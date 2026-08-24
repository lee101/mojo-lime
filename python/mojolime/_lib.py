"""Load the Mojo shared library and describe its C ABI."""

from __future__ import annotations

import ctypes
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
LIB = Path(os.environ.get("MOJOLIME_LIB", ROOT / "dist" / "libmojo-lime.so"))

I = ctypes.c_int64
F = ctypes.c_double

_SIGNATURES = {
    "ml_kernel": ([I, I, I, F], None),
    "ml_affine": ([I, I, I, I, I, I], None),
    "ml_standardize": ([I, I, I, I, I, I], None),
    "ml_affine_standardize": ([I] * 9, None),
    "ml_euclidean_rows": ([I, I, I, I], None),
    "ml_cosine_rows": ([I, I, I, I, F], None),
    "ml_image_neighborhood": ([I, I, I, I, I, I, I, I, I], None),
    "ml_weighted_ridge": ([I] * 11 + [I, I, F, I], I),
}


class BuildError(RuntimeError):
    pass


def _mojo_command() -> list[str]:
    override = os.environ.get("MOJOLIME_MOJO")
    if override:
        return override.split()
    found = shutil.which("mojo")
    if found:
        return [found]
    pixi = shutil.which("pixi")
    if pixi:
        return [pixi, "run", "--manifest-path", str(ROOT / "pixi.toml"), "mojo"]
    raise BuildError("mojo not found; set MOJOLIME_MOJO=/path/to/mojo")


def build(force: bool = False) -> str:
    """Build the shared library when missing or older than its Mojo source."""
    if os.environ.get("MOJOLIME_LIB") and LIB.exists() and not force:
        return str(LIB)
    sources = list(SRC.glob("*.mojo"))
    if not sources:
        if LIB.exists():
            return str(LIB)
        raise BuildError(f"no Mojo sources found under {SRC}")
    if not force and LIB.exists() and LIB.stat().st_mtime >= max(
        source.stat().st_mtime for source in sources
    ):
        return str(LIB)
    LIB.parent.mkdir(parents=True, exist_ok=True)
    command = _mojo_command() + [
        "build",
        "--emit",
        "shared-lib",
        str(SRC / "kernels.mojo"),
        "-o",
        str(LIB),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=1800)
    if result.returncode or not LIB.exists():
        raise BuildError((result.stderr or result.stdout).strip()[:4000])
    return str(LIB)


_loaded: ctypes.CDLL | None = None


def lib() -> ctypes.CDLL:
    global _loaded
    if _loaded is None:
        _loaded = ctypes.CDLL(build())
        for name, (argtypes, restype) in _SIGNATURES.items():
            function = getattr(_loaded, name)
            function.argtypes = argtypes
            function.restype = restype
    return _loaded


def f64(value, *, copy: bool = False) -> np.ndarray:
    raw = np.asarray(value)
    if np.issubdtype(raw.dtype, np.complexfloating):
        raise TypeError("complex values cannot be represented by the float64 ABI")
    if copy:
        return np.array(value, dtype=np.float64, order="C", copy=True)
    return np.ascontiguousarray(value, dtype=np.float64)


def i64(value) -> np.ndarray:
    return np.ascontiguousarray(value, dtype=np.int64)


def addr(value: np.ndarray) -> int:
    return value.ctypes.data


def main() -> int:
    print(build(force="--force" in sys.argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
