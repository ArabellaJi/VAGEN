import os
import sys
import importlib
import importlib.abc


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

        target = "transformers.modeling_utils"

        def should_fallback(exc: ImportError) -> bool:
            message = str(exc)
            return (
                "flash_attn seems to be not installed" in message
                or "Flash Attention 2 is not available" in message
            )

        def force_eager(config) -> None:
            if config is None:
                return
            if hasattr(config, "_attn_implementation"):
                config._attn_implementation = "eager"
            if hasattr(config, "_attn_implementation_internal"):
                config._attn_implementation_internal = "eager"
            print(
                "[sitecustomize] Falling back to eager attention because flash_attn is unavailable",
                file=sys.stderr,
            )

        def patch_module(module) -> None:
            pre_trained_model = getattr(module, "PreTrainedModel", None)
            if pre_trained_model is None:
                return
            if getattr(pre_trained_model, "_vagen_eager_patch_installed", False):
                return

            if hasattr(pre_trained_model, "_check_and_enable_flash_attn_2"):
                original = pre_trained_model._check_and_enable_flash_attn_2.__func__

                def patched_check_and_enable(cls, config, *args, **kwargs):
                    try:
                        return original(cls, config, *args, **kwargs)
                    except ImportError as exc:
                        if not should_fallback(exc):
                            raise
                        force_eager(config)
                        return config

                pre_trained_model._check_and_enable_flash_attn_2 = classmethod(patched_check_and_enable)

            if hasattr(pre_trained_model, "_flash_attn_2_can_dispatch"):
                original_dispatch = pre_trained_model._flash_attn_2_can_dispatch

                def patched_flash_dispatch(self, *args, **kwargs):
                    try:
                        return original_dispatch(self, *args, **kwargs)
                    except ImportError as exc:
                        if not should_fallback(exc):
                            raise
                        force_eager(getattr(self, "config", None))
                        return False

                pre_trained_model._flash_attn_2_can_dispatch = patched_flash_dispatch

            if hasattr(pre_trained_model, "get_correct_attn_implementation"):
                original_get_impl = pre_trained_model.get_correct_attn_implementation

                def patched_get_impl(self, *args, **kwargs):
                    try:
                        return original_get_impl(self, *args, **kwargs)
                    except ImportError as exc:
                        if not should_fallback(exc):
                            raise
                        force_eager(getattr(self, "config", None))
                        return "eager"

                pre_trained_model.get_correct_attn_implementation = patched_get_impl

            if hasattr(pre_trained_model, "_check_and_adjust_attn_implementation"):
                original_adjust_impl = pre_trained_model._check_and_adjust_attn_implementation

                def patched_adjust_impl(self, *args, **kwargs):
                    try:
                        return original_adjust_impl(self, *args, **kwargs)
                    except ImportError as exc:
                        if not should_fallback(exc):
                            raise
                        config = getattr(self, "config", None)
                        force_eager(config)
                        return config

                pre_trained_model._check_and_adjust_attn_implementation = patched_adjust_impl

            pre_trained_model._vagen_eager_patch_installed = True
            print("[sitecustomize] Installed Transformers eager-attention fallback", file=sys.stderr)

        if target in sys.modules:
            patch_module(sys.modules[target])
            return

        class _TransformersModelingUtilsLoader(importlib.abc.Loader):
            def __init__(self, wrapped_loader):
                self._wrapped_loader = wrapped_loader

            def create_module(self, spec):
                if hasattr(self._wrapped_loader, "create_module"):
                    return self._wrapped_loader.create_module(spec)
                return None

            def exec_module(self, module):
                self._wrapped_loader.exec_module(module)
                patch_module(module)

        class _TransformersModelingUtilsFinder(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target_module=None):
                if fullname != target:
                    return None

                original_meta_path = sys.meta_path
                try:
                    sys.meta_path = [finder for finder in original_meta_path if finder is not self]
                    spec = importlib.util.find_spec(fullname)
                finally:
                    sys.meta_path = original_meta_path

                if spec is None or spec.loader is None:
                    return spec

                spec.loader = _TransformersModelingUtilsLoader(spec.loader)
                return spec

        sys.meta_path.insert(0, _TransformersModelingUtilsFinder())
    except Exception as exc:
        print(f"[sitecustomize] Failed to install eager-attention fallback: {exc}", file=sys.stderr)


_install_vagen_sglang_patch()
_install_transformers_eager_attention_fallback()
