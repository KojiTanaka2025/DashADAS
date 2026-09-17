from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QNN_DIR = ROOT / "QNN"
HOST_TRIPLE = "x86_64-linux-clang"


def _looks_like_sdk(path: Path) -> bool:
    return (path / "QNN_README.txt").is_file() or (path / "bin" / "envsetup.sh").is_file()


def find_qnn_sdk() -> Path | None:
    env = os.environ.get("QNN_SDK_ROOT") or os.environ.get("QAIRT_SDK_ROOT") or os.environ.get("DASHADAS_QNN_SDK")
    if env:
        path = Path(env).expanduser().resolve()
        return path if _looks_like_sdk(path) else None

    if not QNN_DIR.is_dir():
        return None

    preferred = os.environ.get("DASHADAS_QNN_VERSION", "").strip()
    if preferred:
        candidate = QNN_DIR / preferred
        return candidate if _looks_like_sdk(candidate) else None

    versions = sorted(
        (child for child in QNN_DIR.iterdir() if child.is_dir() and _looks_like_sdk(child)),
        key=lambda item: item.name,
        reverse=True,
    )
    return versions[0] if versions else None


def qnn_host_libs(sdk: Path) -> Path:
    return sdk / "lib" / HOST_TRIPLE


def qnn_host_bin(sdk: Path) -> Path:
    return sdk / "bin" / HOST_TRIPLE


def qnn_cpu_library(sdk: Path) -> Path:
    return qnn_host_libs(sdk) / "libQnnCpu.so"


def linux_amd64() -> bool:
    return sys.platform.startswith("linux") and platform.machine().lower() in {"x86_64", "amd64"}


def qnn_status() -> dict:
    sdk = find_qnn_sdk()
    if sdk is None:
        return {
            "available": False,
            "usable": False,
            "sdk": None,
            "reason": "Copy a QAIRT/QNN SDK into QNN/<version>/ (see QNN/README.md)",
        }
    if not linux_amd64():
        return {
            "available": True,
            "usable": False,
            "sdk": str(sdk),
            "reason": "QNN CPU emulation needs Linux amd64 (the host .so files are ELF)",
        }
    if not qnn_cpu_library(sdk).is_file():
        return {
            "available": True,
            "usable": False,
            "sdk": str(sdk),
            "reason": f"Missing {qnn_cpu_library(sdk)}",
        }
    if find_converted_model(sdk) is None:
        return {
            "available": True,
            "usable": False,
            "sdk": str(sdk),
            "reason": "Converted QNN model not found. On Linux amd64 run: ./scripts/convert-qnn-yolo.sh",
        }
    return {
        "available": True,
        "usable": True,
        "sdk": str(sdk),
        "reason": None,
    }


def apply_qnn_environment(sdk: Path) -> None:
    os.environ["QNN_SDK_ROOT"] = str(sdk)
    os.environ["QAIRT_SDK_ROOT"] = str(sdk)
    lib = str(qnn_host_libs(sdk))
    bindir = str(qnn_host_bin(sdk))
    py = str(sdk / "lib" / "python")
    os.environ["PATH"] = bindir + os.pathsep + os.environ.get("PATH", "")
    os.environ["LD_LIBRARY_PATH"] = lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    if py not in sys.path:
        sys.path.insert(0, py)


def find_converted_model(sdk: Path | None = None) -> Path | None:
    env = os.environ.get("DASHADAS_QNN_MODEL", "").strip()
    if env:
        path = Path(env).expanduser().resolve()
        return path if path.is_file() else None

    search_roots = [QNN_DIR / "models"]
    if sdk is not None:
        search_roots.append(sdk / "models")
    names = ("libyolo11n.so", "yolo11n.so")
    for root in search_roots:
        for name in names:
            candidate = root / name
            if candidate.is_file():
                return candidate
        if root.is_dir():
            matches = sorted(root.rglob("*yolo11n*.so"))
            if matches:
                return matches[0]
    return None
