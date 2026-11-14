"""M0 environment check. Writes outputs/m00/doctor.json for the M0 checkpoint."""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs" / "m00" / "doctor.json"


def total_mem_gb() -> float:
    try:
        out = subprocess.run(["sysctl", "-n", "hw.memsize"], capture_output=True, text=True)
        return round(int(out.stdout.strip()) / 1024**3, 1)
    except (OSError, ValueError):
        return 0.0


def mps_available() -> bool | None:
    try:
        import torch
    except ImportError:
        return None
    return bool(torch.backends.mps.is_available())


def mlx_available() -> bool | None:
    try:
        import mlx.core  # noqa: F401
    except ImportError:
        return None
    return True


def main() -> int:
    v = sys.version_info
    info = {
        "python": f"{v.major}.{v.minor}.{v.micro}",
        "python_ok": v >= (3, 11),
        "platform": platform.platform(),
        "apple_silicon": sys.platform == "darwin" and platform.machine() == "arm64",
        "total_mem_gb": total_mem_gb(),
        "mps_available": mps_available(),
        "mlx_available": mlx_available(),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(info, indent=2) + "\n")

    def mark(val):
        return {True: "✔", False: "✘", None: "○"}[val]

    print("Environment check")
    print(f"  {mark(info['python_ok'])} Python {info['python']} (need 3.11+)")
    print(f"  {mark(info['apple_silicon'])} Apple Silicon Mac")
    print(f"  {mark(info['total_mem_gb'] > 0)} Memory: {info['total_mem_gb']} GB")
    print(f"  {mark(info['mps_available'])} PyTorch MPS" + (" (installed in M1)" if info["mps_available"] is None else ""))
    print(f"  {mark(info['mlx_available'])} MLX" + (" (installed in M7)" if info["mlx_available"] is None else ""))
    print(f"Saved {OUT.relative_to(ROOT)}\n")
    return 0 if info["python_ok"] else 1


if __name__ == "__main__":
    sys.exit(main())
