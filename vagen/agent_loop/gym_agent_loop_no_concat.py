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
from .agent_loop_no_concat import AgentLoopBase, AgentLoopOutput, register
from verl.utils.profiler import simple_timer
from verl.utils.rollout_trace import rollout_trace_op
from ..envs.gym_image_env import GymImageEnv
from omegaconf import OmegaConf
import traceback
import importlib
logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))
from .gym_agent_loop import (
    convert_obs_to_content,
    extract_success,
    _flatten_text_only_content,
    _normalize_images,
    _tokenize_raw_prompt_for_sglang,
    _trim_multimodal_sequence,
)

class AgentState(Enum):
    PENDING = "pending"
    GENERATING = "generating"
    INTERACTING = "interacting"
    TERMINATED = "terminated"


class AgentData:
    """Container for all mutable trajectory state."""
    def __init__(
        self,
        metrics: Dict[str, Any],
        request_id: str,
        env: GymImageEnv,
        response_limit: int,
        env_name: str,
        sys_msg: Optional[Dict[str, Any]] = None,
        sys_images: Optional[List[Image.Image]] = None,
        cur_msg: Optional[Dict[str, Any]] = None,
        cur_images: Optional[List[Image.Image]] = None,
        group_idx: int = 0,
        traj_idx: int = 0,
        history_window_size: int = 0,
        thumbnail_scale: float = 1.0,
    ):
        self.sys_msg: Optional[Dict[str, Any]] = sys_msg
        self.sys_images: Optional[List[Image.Image]] = sys_images

        self.cur_msg: Optional[Dict[str, Any]] = cur_msg
        self.cur_images: Optional[List[Image.Image]] = cur_images

        # History window config
        # history_window_size: 0=no memory, k>0=keep last k turns, -1=keep all
        self.history_window_size: int = history_window_size
        self.thumbnail_scale: float = thumbnail_scale
        # Each entry: {"obs_msg": Dict, "obs_images": List[Image], "response_text": str}
        self.history_turns: List[Dict[str, Any]] = []
        # Built by _handle_pending_state, used by _handle_generating_state
        self.context_images: List[Image.Image] = []

        self.metrics = metrics
        self.request_id = request_id
        self.env = env
        self.response_limit = response_limit
        self.env_name = env_name
        self.group_idx = group_idx
        self.traj_idx = traj_idx
        # Token buffers
        self.turn_prompt_ids: Optional[List[int]] = None
        self.turn_sglang_prompt_ids: Optional[List[int]] = None
        self.turn_response_ids: Optional[List[int]] = None
        self.turn_response_mask: Optional[List[int]] = None
        self.turn_response_logprobs: Optional[List[int]] = None

        # Env stats
        self.env_turns: int = 0

        # Cached assistant text to step env
        self.last_assistant_text: Optional[str] = None
        self.outputs: List[AgentLoopOutput] = []

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
        cls.history_window_size = config.trainer.get("history_window_size", 0)
        cls.thumbnail_scale = config.trainer.get("thumbnail_scale", 1.0)


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

       
        
        sys_msg={"role": "system", "content": convert_obs_to_content(sys_obs, **kwargs)}
        sys_images=_normalize_images(sys_obs.get("multi_modal_input", {}).get("<image>", []) or [])
        
        cur_msg={"role": "user", "content": convert_obs_to_content(init_obs, **kwargs)}
        cur_images=_normalize_images(init_obs.get("multi_modal_input", {}).get("<image>", []) or [])

        per_turn_response_limit = int(kwargs.get("response_length_per_turn") or self.response_length)
        per_turn_response_limit = min(per_turn_response_limit, self.response_length)
        if per_turn_response_limit <= 0:
            per_turn_response_limit = 1

        agent_data = AgentData(
            sys_msg=sys_msg,
            sys_images=sys_images,
            cur_msg=cur_msg,
            cur_images=cur_images,
            metrics=metrics,
            request_id=request_id,
            env=env,
            response_limit=per_turn_response_limit,
            env_name=kwargs["env_name"],
            group_idx=kwargs["group_idx"],
            traj_idx=kwargs["traj_idx"],
            history_window_size=self.history_window_size,
            thumbnail_scale=self.thumbnail_scale,
        )

        # State machine: always GENERATE -> INTERACT, and decide termination inside INTERACT
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

        # Close env after loop
        await env.close()
        return agent_data.outputs

    def _build_windowed_context(self, agent_data: AgentData):
        """
        Build messages and images for the current turn based on history_window_size.

        history_window_size=0: [sys, cur_obs]              (no memory)
        history_window_size=k: [sys, obs_{n-k}, a_{n-k}, ..., obs_{n-1}, a_{n-1}, cur_obs]
        history_window_size=-1: [sys, all history, cur_obs] (full history in no-concat mode)

        Historical images are resized by thumbnail_scale if < 1.0.
        Returns (messages, image_data).
        """
        messages = [agent_data.sys_msg]
        image_data = list(agent_data.sys_images)

        # Determine which historical turns to include
        history = agent_data.history_turns
        if agent_data.history_window_size == 0:
            window = []
        elif agent_data.history_window_size == -1:
            window = history
        else:
            window = history[-agent_data.history_window_size:]

        scale = agent_data.thumbnail_scale
        for turn in window:
            # Add historical obs (user message)
            messages.append(turn["obs_msg"])
            # Add historical images (optionally thumbnailed)
            for img in turn["obs_images"]:
                if scale < 1.0:
                    new_w = max(1, int(img.width * scale))
                    new_h = max(1, int(img.height * scale))
                    image_data.append(img.resize((new_w, new_h), Image.BILINEAR))
                else:
                    image_data.append(img)
            # Add historical assistant response
            messages.append({"role": "assistant", "content": turn["response_text"]})

        # Add current obs (full resolution)
        messages.append(agent_data.cur_msg)
        image_data.extend(agent_data.cur_images)

        return messages, image_data

    async def _handle_pending_state(self, agent_data: AgentData, sampling_params: Dict[str, Any]) -> AgentState:
        """Encode windowed context (system + history window + current obs) into prompt_ids."""
        messages, image_data = self._build_windowed_context(agent_data)
        # Cache for use in _handle_generating_state
        agent_data.context_images = image_data

        if self.processor is not None:
            raw_prompt = await self.loop.run_in_executor(
                None,
                lambda: self.processor.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    tokenize=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            model_inputs = self.processor(text=[raw_prompt], images=image_data or None, return_tensors="pt")
            agent_data.turn_sglang_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: _tokenize_raw_prompt_for_sglang(self.tokenizer, raw_prompt),
            )
            agent_data.turn_prompt_ids = model_inputs["input_ids"].squeeze(0).tolist()
        else:
            if image_data:
                raise ValueError("Environment returned images but `processor` is None.")
            flat_messages = [_flatten_text_only_content(m) for m in messages]
            agent_data.turn_prompt_ids = await self.loop.run_in_executor(
                None,
                lambda: self.tokenizer.apply_chat_template(
                    flat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=False,
                    **self.apply_chat_template_kwargs,
                ),
            )
            agent_data.turn_sglang_prompt_ids = list(agent_data.turn_prompt_ids)

        if len(agent_data.turn_prompt_ids) > self.prompt_length:
            logger.warning(f"In env:{agent_data.env_name}, prompt length {len(agent_data.turn_prompt_ids)} exceeds prompt_length {self.prompt_length}")
        return AgentState.GENERATING

    
    async def _handle_generating_state(
        self, agent_data: AgentData, sampling_params: Dict[str, Any]
    ) -> AgentState:
        """Generate assistant output and mark generated tokens with mask=1."""
        sampling_params_for_turn = sampling_params.copy()
        max_new_tokens=sampling_params_for_turn.get("max_new_tokens", None) or agent_data.response_limit
        max_new_tokens = min(max_new_tokens, agent_data.response_limit)
        sampling_params_for_turn["max_new_tokens"] = max_new_tokens
        with simple_timer("generate_sequences", agent_data.metrics):
            output = await self.server_manager.generate(
                request_id = agent_data.request_id,
                prompt_ids = agent_data.turn_sglang_prompt_ids,
                sampling_params = sampling_params_for_turn,
                image_data = agent_data.context_images or None,
            )

        agent_data.turn_response_ids = output.token_ids
        agent_data.turn_response_mask = [1] * len(output.token_ids)
        agent_data.turn_prompt_ids += agent_data.turn_response_ids
        agent_data.turn_sglang_prompt_ids += agent_data.turn_response_ids
        if output.log_probs:
            agent_data.turn_response_logprobs = output.log_probs

        # Cache assistant text and add assistant message (text-only)
        assistant_message = await self.loop.run_in_executor(
            None, lambda: self.tokenizer.decode(agent_data.turn_response_ids, skip_special_tokens=True)
        )
        agent_data.last_assistant_text = assistant_message
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

        traj_success = extract_success(info)
        agent_data.env_turns += 1
        last_turn=False
        
        
        
        if done:
            last_turn = True

        if self.env_max_turns is not None and agent_data.env_turns >= int(self.env_max_turns):
            last_turn = True

        
        if len(agent_data.turn_response_mask) >= self.response_length:
            last_turn = True

        # context_images was built by _handle_pending_state (includes windowed history + current)
        turn_images = agent_data.context_images
        
        resp_len = len(agent_data.turn_response_mask)
        response_ids = agent_data.turn_prompt_ids[-resp_len:] if resp_len else []
        prompt_ids = agent_data.turn_prompt_ids[: len(agent_data.turn_prompt_ids) - resp_len]
        prompt_ids, response_ids, response_mask, turn_images, response_logprobs = _trim_multimodal_sequence(
            self.tokenizer,
            prompt_ids,
            response_ids,
            agent_data.turn_response_mask,
            turn_images,
            self.prompt_length,
            self.response_length,
            agent_data.turn_response_logprobs,
        )
        multi_modal_data = {"image": turn_images} if turn_images else {}
        output = AgentLoopOutput(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            response_mask=response_mask,
            multi_modal_data=multi_modal_data,
            response_logprobs=response_logprobs,
            reward_score=float(reward),
            num_turns=1,
            metrics=agent_data.metrics,
            extra_fields={"reward_extra_info": {
                "traj_success": float(traj_success)},
                "image_data": turn_images,
                "last_turn": last_turn,
                "group_idx": agent_data.group_idx,
                "traj_idx": agent_data.traj_idx,
                "turn_idx": agent_data.env_turns,
                          
            },
        )
        agent_data.outputs.append(output)
        
        # Save completed turn to history before updating cur_msg
        # (cur_msg/cur_images = the obs the model just saw; last_assistant_text = its response)
        agent_data.history_turns.append({
            "obs_msg": agent_data.cur_msg,
            "obs_images": list(agent_data.cur_images),
            "response_text": agent_data.last_assistant_text or "",
        })

        # update cur msg and images with the new observation
        cur_msg={"role": "user", "content": convert_obs_to_content(obs, **kwargs)}
        cur_images=_normalize_images(obs.get("multi_modal_input", {}).get("<image>", []) or [])
        agent_data.cur_msg = cur_msg
        agent_data.cur_images = cur_images
        if last_turn:
            return AgentState.TERMINATED

        return AgentState.PENDING
