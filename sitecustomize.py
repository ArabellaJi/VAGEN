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


_install_vagen_sglang_patch()
