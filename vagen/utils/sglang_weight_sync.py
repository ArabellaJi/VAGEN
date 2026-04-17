import inspect
import logging
import os
import shutil
from pathlib import Path
from typing import Any


logger = logging.getLogger(__name__)

ENV_METHOD = "VAGEN_SGLANG_WEIGHT_SYNC_METHOD"
ENV_DIR = "VAGEN_SGLANG_WEIGHT_SYNC_DIR"
ENV_LOAD_FORMAT = "VAGEN_SGLANG_WEIGHT_SYNC_LOAD_FORMAT"
ENV_FLUSH_CACHE = "VAGEN_SGLANG_WEIGHT_SYNC_FLUSH_CACHE"

DEFAULT_METHOD = "tensor"
DEFAULT_LOAD_FORMAT = "auto"
DEFAULT_KEEP_STEPS = 2
DEFAULT_SYNC_DIRNAME = "_sglang_weight_sync"


def _get_nested(config: Any, path: str, default: Any = None) -> Any:
    current = config
    for part in path.split("."):
        if current is None:
            return default
        try:
            if hasattr(current, "get"):
                sentinel = object()
                value = current.get(part, sentinel)
                if value is not sentinel:
                    current = value
                    continue
        except Exception:
            pass
        try:
            current = getattr(current, part)
            continue
        except Exception:
            return default
    return current if current is not None else default


def _normalize_method(method: str | None) -> str:
    return (method or DEFAULT_METHOD).strip().lower()


def _get_env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def get_sglang_weight_sync_method(config: Any = None) -> str:
    method = os.getenv(ENV_METHOD)
    if method:
        return _normalize_method(method)
    if config is not None:
        config_method = _get_nested(config, "actor_rollout_ref.rollout.weight_sync.method")
        if config_method:
            return _normalize_method(str(config_method))
    return DEFAULT_METHOD


def get_sglang_weight_sync_root(config: Any = None, default_local_dir: str | None = None) -> str:
    root = os.getenv(ENV_DIR)
    if root:
        return root

    if config is not None:
        configured_root = _get_nested(config, "actor_rollout_ref.rollout.weight_sync.model_dir")
        if configured_root:
            return str(configured_root)

    base_dir = default_local_dir
    if base_dir is None and config is not None:
        base_dir = _get_nested(config, "trainer.default_local_dir")
    if not base_dir:
        raise ValueError(
            "SGLang disk weight sync requires either "
            f"{ENV_DIR} or trainer.default_local_dir to be set."
        )
    return str(Path(base_dir) / DEFAULT_SYNC_DIRNAME)


def is_sglang_disk_weight_sync_enabled(config: Any = None) -> bool:
    if get_sglang_weight_sync_method(config) != "disk":
        return False
    rollout_name = str(_get_nested(config, "actor_rollout_ref.rollout.name", ""))
    rollout_mode = str(_get_nested(config, "actor_rollout_ref.rollout.mode", ""))
    return rollout_name == "sglang" and rollout_mode == "async"


def ensure_sglang_disk_weight_sync_env(config: Any) -> dict[str, str]:
    if not is_sglang_disk_weight_sync_enabled(config):
        return {}

    sync_root = get_sglang_weight_sync_root(config=config)
    load_format = os.getenv(ENV_LOAD_FORMAT) or str(
        _get_nested(config, "actor_rollout_ref.rollout.weight_sync.load_format", DEFAULT_LOAD_FORMAT)
    )
    env = {
        ENV_METHOD: "disk",
        ENV_DIR: sync_root,
        ENV_LOAD_FORMAT: load_format,
    }
    return env


def get_sglang_sync_step_dir(sync_root: str, global_step: int) -> str:
    return str(Path(sync_root) / f"global_step_{int(global_step)}")


def get_sglang_actor_sync_dir(sync_root: str, global_step: int) -> str:
    return str(Path(get_sglang_sync_step_dir(sync_root, global_step)) / "actor")


def get_sglang_hf_model_dir(sync_root: str, global_step: int) -> str:
    return str(Path(get_sglang_actor_sync_dir(sync_root, global_step)) / "huggingface")


def get_sglang_latest_step_file(sync_root: str) -> str:
    return str(Path(sync_root) / "latest_step.txt")


def update_latest_sync_step(sync_root: str, global_step: int) -> None:
    sync_path = Path(sync_root)
    sync_path.mkdir(parents=True, exist_ok=True)
    latest_file = Path(get_sglang_latest_step_file(sync_root))
    tmp_file = latest_file.with_suffix(".tmp")
    tmp_file.write_text(str(int(global_step)), encoding="utf-8")
    os.replace(tmp_file, latest_file)


def read_latest_sync_step(sync_root: str) -> int | None:
    latest_file = Path(get_sglang_latest_step_file(sync_root))
    if not latest_file.exists():
        return None
    value = latest_file.read_text(encoding="utf-8").strip()
    if not value:
        return None
    return int(value)


def prune_old_sync_steps(sync_root: str, keep: int = DEFAULT_KEEP_STEPS) -> None:
    sync_path = Path(sync_root)
    if not sync_path.exists():
        return

    step_dirs: list[tuple[int, Path]] = []
    for child in sync_path.iterdir():
        if not child.is_dir():
            continue
        prefix = "global_step_"
        if not child.name.startswith(prefix):
            continue
        try:
            step = int(child.name[len(prefix) :])
        except ValueError:
            continue
        step_dirs.append((step, child))

    step_dirs.sort()
    for _, old_dir in step_dirs[:-keep]:
        shutil.rmtree(old_dir, ignore_errors=True)


def apply_sglang_disk_weight_sync_monkey_patch(force: bool = False) -> bool:
    if not force and get_sglang_weight_sync_method() != "disk":
        return False

    try:
        from verl.workers.rollout.sglang_rollout.http_server_engine import (
            AsyncHttpServerAdapter,
            HttpServerAdapter,
        )
        from verl.workers.rollout.sglang_rollout.sglang_rollout import ServerAdapter
    except Exception as exc:
        logger.warning("Failed to import SGLang rollout modules for disk sync patch: %s", exc)
        return False

    if not hasattr(HttpServerAdapter, "update_weights_from_disk"):
        def _update_weights_from_disk(
            self,
            model_path: str,
            load_format: str | None = None,
            flush_cache: bool = True,
        ) -> dict[str, Any]:
            payload = {"model_path": model_path, "flush_cache": flush_cache}
            if load_format is not None:
                payload["load_format"] = load_format
            return self._make_request("update_weights_from_disk", payload)

        HttpServerAdapter.update_weights_from_disk = _update_weights_from_disk

    if not hasattr(AsyncHttpServerAdapter, "update_weights_from_disk"):
        async def _async_update_weights_from_disk(
            self,
            model_path: str,
            load_format: str | None = None,
            flush_cache: bool = True,
        ) -> dict[str, Any]:
            payload = {"model_path": model_path, "flush_cache": flush_cache}
            if load_format is not None:
                payload["load_format"] = load_format
            return await self._make_async_request("update_weights_from_disk", payload)

        AsyncHttpServerAdapter.update_weights_from_disk = _async_update_weights_from_disk

    if getattr(ServerAdapter.update_weights, "_vagen_disk_sync_patched", False):
        return True

    original_update_weights = ServerAdapter.update_weights

    async def _patched_update_weights(self, weights, global_steps: int = None, **kwargs):
        import os as _os
        print(
            f"[disk_sync] _patched_update_weights CALLED "
            f"pid={_os.getpid()} global_steps={global_steps} "
            f"sync_root={_os.getenv(ENV_DIR)!r}",
            flush=True,
        )

        peft_config = kwargs.get("peft_config")
        if peft_config is not None:
            return await original_update_weights(self, weights, global_steps=global_steps, **kwargs)

        await self._init_server_adapter()
        if self.device_mesh["infer_tp"].get_local_rank() != 0:
            return

        sync_root = os.getenv(ENV_DIR)
        if not sync_root:
            raise RuntimeError(
                f"{ENV_DIR} must be set when {ENV_METHOD}=disk so SGLang can reload weights from disk."
            )

        target_step = global_steps
        if target_step is None:
            target_step = read_latest_sync_step(sync_root)

        if target_step is None:
            raise RuntimeError(
                "No exported rollout checkpoint is available yet. "
                "Expected latest_step.txt in the SGLang disk sync directory."
            )

        model_path = get_sglang_hf_model_dir(sync_root, int(target_step))
        if not os.path.isdir(model_path):
            latest_step = read_latest_sync_step(sync_root)
            if latest_step is not None and latest_step != target_step:
                model_path = get_sglang_hf_model_dir(sync_root, int(latest_step))
                target_step = latest_step

        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                "SGLang disk weight sync could not find a Hugging Face checkpoint at "
                f"{model_path}."
            )

        import time as _time
        load_format = os.getenv(ENV_LOAD_FORMAT, DEFAULT_LOAD_FORMAT)
        flush_cache = _get_env_bool(ENV_FLUSH_CACHE, True)
        print(
            f"[disk_sync] calling update_weights_from_disk "
            f"model_path={model_path} load_format={load_format} flush_cache={flush_cache}",
            flush=True,
        )
        _t = _time.time()
        update_result = self._engine.update_weights_from_disk(
            model_path=model_path,
            load_format=load_format,
            flush_cache=flush_cache,
        )
        if inspect.isawaitable(update_result):
            await update_result
        print(f"[disk_sync] update_weights_from_disk DONE in {_time.time()-_t:.1f}s", flush=True)
        if global_steps is not None:
            await self.server_actor.set_global_steps.remote(global_steps)

    _patched_update_weights._vagen_disk_sync_patched = True
    ServerAdapter.update_weights = _patched_update_weights
    logger.info("Installed VAGEN SGLang disk weight sync monkey patch")
    return True
