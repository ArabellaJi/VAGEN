import os
import sys


def _install_vagen_sglang_patch() -> None:
    method = os.getenv("VAGEN_SGLANG_WEIGHT_SYNC_METHOD", "").strip().lower()
    if method != "disk":
        return

    try:
        from vagen.utils.sglang_weight_sync import apply_sglang_disk_weight_sync_monkey_patch

        applied = apply_sglang_disk_weight_sync_monkey_patch()
        if applied:
            print("[sitecustomize] Installed VAGEN SGLang disk weight sync patch", file=sys.stderr)
    except Exception as exc:
        print(f"[sitecustomize] Failed to install VAGEN SGLang disk sync patch: {exc}", file=sys.stderr)


def _install_transformers_eager_attention_fallback() -> None:
    flag = os.getenv("VAGEN_FORCE_EAGER_ATTN", "").strip().lower()
    if flag not in {"1", "true", "yes", "on"}:
        return

    try:
        import importlib.util

        if importlib.util.find_spec("flash_attn") is not None:
            return

        from transformers.modeling_utils import PreTrainedModel

        original = PreTrainedModel._check_and_enable_flash_attn_2.__func__

        def patched(cls, config, *args, **kwargs):
            try:
                return original(cls, config, *args, **kwargs)
            except ImportError as exc:
                message = str(exc)
                should_fallback = (
                    "flash_attn seems to be not installed" in message
                    or "Flash Attention 2 is not available" in message
                )
                if not should_fallback:
                    raise

                if hasattr(config, "_attn_implementation"):
                    config._attn_implementation = "eager"
                if hasattr(config, "_attn_implementation_internal"):
                    config._attn_implementation_internal = "eager"
                print(
                    "[sitecustomize] Falling back to eager attention because flash_attn is unavailable",
                    file=sys.stderr,
                )
                return config

        PreTrainedModel._check_and_enable_flash_attn_2 = classmethod(patched)
        print("[sitecustomize] Installed Transformers eager-attention fallback", file=sys.stderr)
    except Exception as exc:
        print(f"[sitecustomize] Failed to install eager-attention fallback: {exc}", file=sys.stderr)


_install_vagen_sglang_patch()
_install_transformers_eager_attention_fallback()
