# Copyright 2025 Bytedance Ltd.
# Licensed under the Apache License, Version 2.0

import asyncio
import logging
import os
import re
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import uuid4

from PIL import Image
from verl.experimental.agent_loop.agent_loop import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

def _flatten_text_only_content(msg):
    """
    convert message['content'] from multimodal list to plain text
    - only allow type == 'text'
    - concatenate multiple text blocks in order
    """
    content = msg.get("content")

    if isinstance(content, str):
        return msg

    if not isinstance(content, list):
        raise TypeError(f"Unsupported content type: {type(content)}")

    texts = []
    for block in content:
        if not isinstance(block, dict):
            raise TypeError(f"Invalid content block: {block}")

        block_type = block.get("type")
        if block_type != "text":
            raise AssertionError(
                f"Non-text block found in text-only tokenizer path: {block_type}"
            )
        texts.append(block.get("text", ""))

    new_msg = dict(msg)
    new_msg["content"] = "".join(texts)
    return new_msg


class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        messages: List[Dict[str, Any]],
        image_data: List[Image.Image],
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
    ):
        self.messages = messages
        self.image_data = image_data
        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name

        # Token buffers
        self.prompt_ids: List[int] = []
        self.sglang_prompt_ids: List[int] = []
        self.response_ids: List[int] = []
        self.response_mask: List[int] = []
        self.response_logprobs: List[float] = []

        # Env stats
        self.env_rewards: List[float] = []
        self.traj_success: bool = False
        self.env_turns: int = 0

        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None

        # MEM1: compact memory state (managed by agent loop, not env)
        self.use_mem1: bool = False
        self.memory_state: str = "me=(0,0) step=0"


# -------------------- MM helpers --------------------

def _normalize_images(imgs: List[Image.Image]) -> List[Image.Image]:
    """Ensure PIL RGB and drop Nones."""
    out: List[Image.Image] = []
    for im in imgs or []:
        if im is None:
            continue
        out.append(im.convert("RGB") if isinstance(im, Image.Image) else im)
    return out

def _tokenize_raw_prompt_for_sglang(tokenizer, raw_prompt: str) -> List[int]:
    """Tokenize a chat-template string without expanding image placeholders."""
    return tokenizer(raw_prompt, add_special_tokens=False)["input_ids"]

def _token_id(tokenizer, token: str) -> Optional[int]:
    token_id = tokenizer.convert_tokens_to_ids(token)
    return token_id if isinstance(token_id, int) and token_id >= 0 else None

def _find_image_spans(tokenizer, token_ids: List[int]) -> List[tuple[int, int]]:
    """Return half-open spans covering each expanded Qwen image token block."""
    image_id = _token_id(tokenizer, "<|image_pad|>")
    if image_id is None:
        return []

    vision_start_id = _token_id(tokenizer, "<|vision_start|>")
    vision_end_id = _token_id(tokenizer, "<|vision_end|>")
    spans: List[tuple[int, int]] = []
    i = 0
    n = len(token_ids)
    while i < n:
        if token_ids[i] != image_id:
            i += 1
            continue

        pad_start = i
        while i < n and token_ids[i] == image_id:
            i += 1
        pad_end = i

        span_start = pad_start
        if vision_start_id is not None and pad_start > 0 and token_ids[pad_start - 1] == vision_start_id:
            span_start = pad_start - 1

        span_end = pad_end
        if vision_end_id is not None and pad_end < n and token_ids[pad_end] == vision_end_id:
            span_end = pad_end + 1

        spans.append((span_start, span_end))
    return spans

def _trim_multimodal_sequence(
    tokenizer,
    prompt_ids: List[int],
    response_ids: List[int],
    response_mask: List[int],
    images: List[Image.Image],
    prompt_length: int,
    response_length: int,
    response_logprobs: Optional[List[float]] = None,
):
    """Trim token windows and keep only images whose token spans remain present."""
    full_ids = list(prompt_ids) + list(response_ids)
    prompt_end = len(prompt_ids)
    window_start = max(0, prompt_end - prompt_length)
    window_end = prompt_end + min(len(response_ids), response_length)
    spans = _find_image_spans(tokenizer, full_ids)

    for span_start, span_end in spans:
        if span_start < window_start < span_end:
            window_start = span_end
        if span_start < window_end < span_end:
            window_end = span_start
            break

    kept_images: List[Image.Image] = []
    for idx, (span_start, span_end) in enumerate(spans):
        if idx >= len(images):
            break
        if window_start <= span_start and span_end <= window_end:
            kept_images.append(images[idx])

    if images and len(spans) != len(images):
        logger.warning("Image span/image count mismatch before trimming: spans=%d images=%d", len(spans), len(images))

    trimmed_prompt_ids = full_ids[window_start:prompt_end]
    response_keep = max(0, window_end - prompt_end)
    trimmed_response_ids = response_ids[:response_keep]
    trimmed_response_mask = response_mask[:response_keep]
    trimmed_response_logprobs = response_logprobs[:response_keep] if response_logprobs is not None else None

    return trimmed_prompt_ids, trimmed_response_ids, trimmed_response_mask, kept_images, trimmed_response_logprobs

def extract_success(info: Dict[str, Any], success_keys: str = "success|is_success") -> bool:
    """Extract success flag from env info dict."""
    for key in success_keys.split("|"):
        if key in info:
            return bool(info[key])
    return False

def convert_obs_to_content(
    obs: Dict[str, Any],
    obs_text_key: str = "obs_str",
    image_placeholder: str = "<image>",
    video_placeholder: str = "<video>",
    multi_modal_key: str = "multi_modal_input",
    **kwargs,
) -> List[Dict[str, Any]]:
    """Convert obs['obs_str'] containing <image>/<video> into structured content."""
    text = obs[obs_text_key]
    mmi = obs.get(multi_modal_key, {}) or {}

    # Simple strict consistency check
    num_img_tok = text.count(image_placeholder)
    num_vid_tok = text.count(video_placeholder)
    num_imgs = len(mmi.get(image_placeholder, []) or [])
    num_vids = len(mmi.get(video_placeholder, []) or [])
    assert num_img_tok == num_imgs, f"#images ({num_imgs}) != #{image_placeholder} ({num_img_tok})"
    assert num_vid_tok == num_vids, f"#videos ({num_vids}) != #{video_placeholder} ({num_vid_tok})"

    # Split and keep tokens
    pattern = f"({re.escape(image_placeholder)}|{re.escape(video_placeholder)})"
    segments = re.split(pattern, text)

    content: List[Dict[str, Any]] = []
    for seg in segments:
        if not seg:
            continue
        if seg == image_placeholder:
            content.append({"type": "image"})
        elif seg == video_placeholder:
            content.append({"type": "video"})
        else:
            content.append({"type": "text", "text": seg})
    return content


def _extract_memory(text: str, default: str = "me=(0,0) step=0") -> str:
    """Extract <memory>...</memory> content from a model response."""
    m = re.search(r"<memory>(.*?)</memory>", text, re.DOTALL)
    return m.group(1).strip() if m else default


def _build_mem1_user_text(memory_state: str, obs_str: str) -> str:
    """Build the user message text for a MEM1 step."""
    return f"[Your memory state]\n{memory_state}\n\n{obs_str}"


# -------------------- Gym Agent Loop --------------------

class GymAgentLoop(AgentLoopBase):
    @classmethod
    def init_class(cls, config, tokenizer, processor, **kwargs):
        if cls._class_initialized:
            return
        cls._class_initialized = True
        print("Performing class-level GymAgentLoop initialization")

        cls.tokenizer = tokenizer
        cls.processor = processor
        cls.multi_turn_cfg = config.actor_rollout_ref.rollout.multi_turn
        
        # Store module paths for lazy loading; environments are imported on first use
        cls.env_registry_paths = dict(config.env_registry.items())
        cls.env_registry = {}
            
        cls.apply_chat_template_kwargs = config.data.get("apply_chat_template_kwargs", {})
        cls.prompt_length = config.actor_rollout_ref.rollout.prompt_length
        cls.response_length = config.actor_rollout_ref.rollout.response_length
      
        _placeholder = [{"role": "system", "content": "placeholder"}]
        if processor is not None:
            _prefix_text = processor.apply_chat_template(
                _placeholder, add_generation_prompt=False, tokenize=False, **cls.apply_chat_template_kwargs
            )
            cls.system_prompt_prefix = processor(text=[_prefix_text], return_tensors="pt")["input_ids"].squeeze(0).tolist()
            cls.system_prompt_prefix_sglang = _tokenize_raw_prompt_for_sglang(tokenizer, _prefix_text)
        else:
            cls.system_prompt_prefix = tokenizer.apply_chat_template(
                _placeholder, add_generation_prompt=False, tokenize=True, return_dict=False, **cls.apply_chat_template_kwargs
            )
            cls.system_prompt_prefix_sglang = cls.system_prompt_prefix

    @rollout_trace_op
    async def run(self, sampling_params: Dict[str, Any], **kwargs) -> AgentLoopOutput:
        metrics: Dict[str, Any] = {}
        request_id = uuid4().hex

        # Build env (lazy import on first use)
        env_name = kwargs["env_name"]
        if env_name not in self.env_registry:
            if env_name not in self.env_registry_paths:
                raise KeyError(f"Unknown env: {env_name}. Available: {list(self.env_registry_paths.keys())}")
            module_path, class_name = self.env_registry_paths[env_name].rsplit(".", 1)
            module = importlib.import_module(module_path)
            self.env_registry[env_name] = getattr(module, class_name)
        env_cls = self.env_registry[env_name]
        env_config = kwargs["config"]
        seed = kwargs["seed"]
        self.env_max_turns = kwargs.get("max_turns", None)
        env: GymImageEnv = env_cls(env_config=env_config)

        # Bootstrap: reset -> system_prompt (message order: system, then initial user)
        init_obs, info = await env.reset(seed=seed)
        sys_obs = await env.system_prompt()

        messages: List[Dict[str, Any]] = []
        image_data: List[Image.Image] = []

        # Detect MEM1 mode from env config
        env_config = kwargs.get("config", {})
        _pf = env_config.get("prompt_format", "") if isinstance(env_config, dict) else getattr(env_config, "prompt_format", "")
        use_mem1 = (_pf == "mem1")

        if sys_obs:
            messages.append({"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)})
            sys_imgs = sys_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(sys_imgs))
        if init_obs:
            if use_mem1:
                # Prepend empty memory state to initial observation
                init_obs_mem1 = dict(init_obs)
                init_obs_mem1["obs_str"] = _build_mem1_user_text("me=(0,0) step=0", init_obs.get("obs_str", ""))
                messages.append({"role": "user", "content": convert_obs_to_content(init_obs_mem1, **kwargs)})
            else:
                messages.append({"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)})
            init_imgs = init_obs.get("multi_modal_input", {}).get("<image>", []) or []
            image_data.extend(_normalize_images(init_imgs))

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            messages=messages,
            image_data=image_data,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
        )
        agent_data.use_mem1 = use_mem1

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
        try:
            state = AgentState.PENDING
            while state != AgentState.TERMINATED:
                if state == AgentState.PENDING:
                    state = await self._handle_pending_state(agent_data, sampling_params)
                elif state == AgentState.GENERATING:
                    state = await self._handle_generating_state(agent_data, sampling_params)
                elif state == AgentState.INTERACTING:
                    state = await self._handle_env_state(agent_data, **kwargs)
                else:
                    logger.error(f"Invalid state: {state}")
                    state = AgentState.TERMINATED
        finally:
            await env.close()

        # Finalize output
        resp_len = len(agent_data.response_mask)
        response_ids = agent_data.prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.prompt_ids[: len(agent_data.prompt_ids) - resp_len]
        prompt_ids, response_ids, response_mask, image_data, response_logprobs = _trim_multimodal_sequence(
            self.tokenizer,
            prompt_ids,
            response_ids,
            agent_data.response_mask,
            agent_data.image_data,
            self.prompt_length,
            self.response_length,
            agent_data.response_logprobs if agent_data.response_logprobs else None,
        )
        multi_modal_data = {"image": image_data} if image_data else {}

        if len(prompt_ids) > self.prompt_length:
            logger.warning(
                f"In env:{agent_data.env_name}, prompt_ids length {len(prompt_ids)} exceeds prompt_length {self.prompt_length}",
            )
        if len(response_ids) > self.response_length:
            logger.warning(
                f"In env:{agent_data.env_name}, response_ids length {len(response_ids)} exceeds response_length {self.response_length}",
            )

        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs,
            reward_score=sum(agent_data.env_rewards) if agent_data.env_rewards else 0.0,
            num_turns=agent_data.env_turns,
            metrics=agent_data.metrics,
            extra_fields={ "image_data": image_data,"reward_extra_info": {"traj_success": float(agent_data.traj_success)}},
        )
        return output

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode initial (system + first user) messages into prompt_ids."""
        if self.processor is not None and agent_data.image_data:
            raw_prompt = self.processor.apply_chat_template(
                agent_data.messages,
                add_generation_prompt=True,
                tokenize=False,
                **self.apply_chat_template_kwargs,
            )
            model_inputs = self.processor(text=[raw_prompt], images=agent_data.image_data, return_tensors="pt")
            agent_data.sglang_prompt_ids = _tokenize_raw_prompt_for_sglang(self.tokenizer, raw_prompt)
            agent_data.prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        else:
            if agent_data.image_data:
                raise ValueError("Environment returned images but `processor` is None.")

            flat_messages = [_flatten_text_only_content(msg) for msg in agent_data.messages]
            agent_data.prompt_ids = self.tokenizer.apply_chat_template(
                flat_messages,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.apply_chat_template_kwargs,
            )
            agent_data.sglang_prompt_ids = list(agent_data.prompt_ids)
        
        if len(agent_data.prompt_ids)>self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, initial prompt length {len(agent_data.prompt_ids)} exceeds prompt_length {self.prompt_length}")
        if os.getenv("VAGEN_DEBUG_GENERATE") == "1":
            _dec_p = self.tokenizer.decode(agent_data.prompt_ids, skip_special_tokens=False)
            logger.warning(
                "[VAGEN_DEBUG] PENDING env=%s n_prompt_ids=%d decoded_tail=%r",
                agent_data.env_name, len(agent_data.prompt_ids), _dec_p[-300:],
            )
        return AgentState.GENERATING

    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
            

        _debug = os.getenv("VAGEN_DEBUG_GENERATE") == "1"
        if _debug:
            _dec = self.tokenizer.decode(agent_data.sglang_prompt_ids, skip_special_tokens=False)
            logger.warning(
                "[VAGEN_DEBUG] turn=%d env=%s n_sglang_ids=%d decoded_head=%r",
                agent_data.env_turns, agent_data.env_name,
                len(agent_data.sglang_prompt_ids), _dec[:300],
            )
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id=agent_data.request_id,
                prompt_ids=agent_data.sglang_prompt_ids,
                sampling_params=sampling_params_for_turn,
                image_data=agent_data.image_data or None,
            )
        if _debug:
            logger.warning(
                "[VAGEN_DEBUG] turn=%d response_ids_head=%s decoded_resp=%r",
                agent_data.env_turns,
                (output.token_ids or [])[:15],
                self.tokenizer.decode((output.token_ids or [])[:40], skip_special_tokens=False),
            )

        agent_data.response_ids = output.token_ids
        if len(output.token_ids)>agent_data.response_limit:
            logger.warning(f"In env:{agent_data.env_name}, generated response length {len(output.token_ids)} exceeds per-turn response_limit {agent_data.response_limit}")
        agent_data.prompt_ids += agent_data.response_ids
        agent_data.sglang_prompt_ids += agent_data.response_ids
        agent_data.response_mask += [1] * len(agent_data.response_ids)
        if output.log_probs:
            agent_data.response_logprobs += output.log_probs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = self.tokenizer.decode(agent_data.response_ids, skip_special_tokens=True)
        agent_data.last_assistant_text = assistant_message
        agent_data.messages.append({"role": "assistant", "content": assistant_message})

        # MEM1: extract and cache new memory state from response
        if agent_data.use_mem1:
            agent_data.memory_state = _extract_memory(assistant_message, default=agent_data.memory_state)

        return AgentState.INTERACTING

    async def _handle_env_state(self, agent_data: AgentData, **kwargs) -> AgentState:
        """
        Step the environment with last assistant action; always collect reward first.
        If terminal (done/success/turn-limit/token-limit), stop WITHOUT appending user suffix,
        so the episode ends on an assistant turn.
        """
        action_str = agent_data.last_assistant_text or ""
        try:
            obs, reward, done, info = await agent_data.env.step(action_str)
            # traceback
        except Exception as exc:
            logger.error(
                "Environment step failed in '%s' with action %r: %s",
                agent_data.env_name,
                action_str,
                exc,
            )
            logger.error("Environment traceback:\n%s", traceback.format_exc())
            obs, reward, done, info = {"obs_str":"Environment Error"}, 0.0, True, {"traj_success": False}

        agent_data.env_rewards.append(float(reward))
        agent_data.traj_success = extract_success(info)
        agent_data.env_turns += 1
        # Termination rule #3: env done or success
        if done or agent_data.traj_success:
            return AgentState.TERMINATED

        # Termination rule #2: env turn-limit (if set)
        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            return AgentState.TERMINATED

        # Termination rule #1: response token-limit
        if len(agent_data.response_mask) >= self.response_length:
            return AgentState.TERMINATED

        # Not terminal -> append user suffix for next turn
        new_images = obs.get("multi_modal_input", {}).get("<image>", []) or []
        new_images = _normalize_images(new_images)
        _placeholder = {"role": "system", "content": "placeholder"}

        if agent_data.use_mem1:
            # MEM1: inject memory state into user message
            obs_mem1 = dict(obs)
            obs_mem1["obs_str"] = _build_mem1_user_text(agent_data.memory_state, obs.get("obs_str", ""))
            user_content = convert_obs_to_content(obs_mem1, **kwargs)
            user_msg = {"role": "user", "content": user_content}
            agent_data.messages.append(user_msg)

            if new_images:
                raise ValueError("MEM1 mode does not support vision observations.")

            flat_user_msg = _flatten_text_only_content(user_msg)

            # prompt_ids (training): append suffix as usual
            suffix_ids = self.tokenizer.apply_chat_template(
                [_placeholder, flat_user_msg],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.apply_chat_template_kwargs,
            )
            suffix_ids = suffix_ids[len(self.system_prompt_prefix):]
            agent_data.prompt_ids += suffix_ids
            agent_data.response_mask += [0] * len(suffix_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(suffix_ids)

            # sglang_prompt_ids (inference): RESET to fresh [sys + mem+obs]
            # so the model only sees memory + current obs, not full history
            fresh_msgs = [agent_data.messages[0], user_msg]
            flat_fresh = [_flatten_text_only_content(m) for m in fresh_msgs]
            fresh_ids = self.tokenizer.apply_chat_template(
                flat_fresh,
                add_generation_prompt=True,
                tokenize=True,
                return_dict=False,
                **self.apply_chat_template_kwargs,
            )
            agent_data.sglang_prompt_ids = fresh_ids  # full reset

        else:
            user_content = convert_obs_to_content(obs, **kwargs)
            user_msg = {"role": "user", "content": user_content}
            agent_data.messages.append(user_msg)

            if self.processor is not None and new_images:
                raw_user_suffix = self.processor.apply_chat_template(
                    [_placeholder, user_msg],
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                )
                model_inputs = self.processor(text=[raw_user_suffix], images=new_images, return_tensors="pt")
                sglang_response_ids = _tokenize_raw_prompt_for_sglang(self.tokenizer, raw_user_suffix)
                response_ids = model_inputs["input_ids"].squeeze(0).tolist()
            else:
                if new_images and self.processor is None:
                    raise ValueError("Environment returned images but `processor` is None.")
                flat_user_msg = _flatten_text_only_content(user_msg)
                response_ids = self.tokenizer.apply_chat_template(
                    [_placeholder, flat_user_msg],
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                )
                sglang_response_ids = response_ids
            response_ids = response_ids[len(self.system_prompt_prefix):]
            sglang_response_ids = sglang_response_ids[len(self.system_prompt_prefix_sglang):]
            agent_data.prompt_ids += response_ids
            agent_data.sglang_prompt_ids += sglang_response_ids
            agent_data.response_mask += [0] * len(response_ids)
            if agent_data.response_logprobs:
                agent_data.response_logprobs += [0.0] * len(response_ids)

            if new_images:
                agent_data.image_data.extend(new_images)

        return AgentState.GENERATING
