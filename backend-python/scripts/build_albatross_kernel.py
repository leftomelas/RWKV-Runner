import argparse
import json
import os
from pathlib import Path
import shutil
import sys

import torch

BACKEND_ROOT = Path(__file__).resolve().parent.parent
if str(BACKEND_ROOT) not in sys.path:
    sys.path.insert(0, str(BACKEND_ROOT))

from albatross.kernel_loader import (
    current_platform_tag,
    current_python_abi,
    extension_build_options,
    validate_python_extension_build_environment,
)


KERNEL_NAME = "rwkv7_state_fwd_fp16"
HEAD_SIZE = 64


def parse_arches(raw_arches: str) -> list[str]:
    arches = []
    for raw_arch in raw_arches.replace(";", ",").split(","):
        arch = raw_arch.strip().lower()
        if not arch:
            continue
        if arch.startswith("sm"):
            digits = arch[2:]
        elif "." in arch:
            major, minor = arch.split(".", 1)
            digits = f"{major}{minor}"
        else:
            digits = arch
        if not digits.isdigit() or len(digits) < 2:
            raise ValueError(f"Invalid CUDA arch: {raw_arch}")
        normalized = f"sm{digits}"
        if normalized not in arches:
            arches.append(normalized)
    if not arches:
        raise ValueError("At least one CUDA arch is required")
    return arches


def torch_cuda_arch_list(arches: list[str]) -> str:
    converted = []
    for arch in arches:
        digits = arch[2:] if arch.startswith("sm") else arch
        converted.append(f"{digits[:-1]}.{digits[-1]}")
    return ";".join(converted)


def library_suffix() -> str:
    return ".pyd" if os.name == "nt" else ".so"


def find_built_library(build_dir: Path) -> Path:
    candidates = sorted(
        build_dir.glob(f"*{library_suffix()}"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for candidate in candidates:
        if KERNEL_NAME in candidate.name:
            return candidate
    if candidates:
        return candidates[0]
    raise FileNotFoundError(f"No built extension library found in {build_dir}")


def update_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"kernels": []}

    def same_context(candidate: dict) -> bool:
        keys = ["name", "torch", "cuda", "python_abi", "platform"]
        return all(candidate.get(key) == entry.get(key) for key in keys)

    kernels = [
        candidate
        for candidate in manifest.get("kernels", [])
        if not same_context(candidate)
    ]
    kernels.append(entry)
    manifest["kernels"] = kernels
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_kernel(args) -> Path:
    from torch.utils.cpp_extension import load

    if args.python_include:
        os.environ["ALBATROSS_PYTHON_INCLUDE"] = args.python_include
    if args.python_lib_dir:
        os.environ["ALBATROSS_PYTHON_LIB_DIR"] = args.python_lib_dir

    validate_python_extension_build_environment()
    arches = parse_arches(args.arch)
    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list(arches)

    repo_backend = Path(__file__).resolve().parent.parent
    source_root = (repo_backend / args.source_root).resolve()
    output_root = (repo_backend / args.output_root).resolve()
    build_dir = (repo_backend / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)

    sources = [
        str(source_root / "cuda" / f"{KERNEL_NAME}.cpp"),
        str(source_root / "cuda" / f"{KERNEL_NAME}.cu"),
    ]
    extension_options = extension_build_options()
    load(
        name=KERNEL_NAME,
        sources=sources,
        is_python_module=False,
        build_directory=str(build_dir),
        verbose=True,
        extra_cuda_cflags=[
            "-res-usage",
            "--use_fast_math",
            "-O3",
            "--extra-device-vectorization",
            f"-D_N_={HEAD_SIZE}",
        ]
        + (["-Xptxas -O3"] if os.name != "nt" else []),
        **extension_options,
    )

    built_library = find_built_library(build_dir)
    relative_library = (
        Path(f"torch-{torch.__version__}")
        / current_platform_tag()
        / current_python_abi()
        / f"{KERNEL_NAME}{library_suffix()}"
    )
    output_library = output_root / relative_library
    output_library.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(built_library, output_library)

    update_manifest(
        output_root / "manifest.json",
        {
            "name": KERNEL_NAME,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "python_abi": current_python_abi(),
            "platform": current_platform_tag(),
            "arch": arches,
            "path": relative_library.as_posix(),
            "source": "local-build",
        },
    )
    return output_library


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build and register the Albatross RWKV-7 CUDA extension."
    )
    parser.add_argument("--arch", default="sm80,sm86,sm89,sm90,sm120")
    parser.add_argument("--output-root", default="albatross/kernels")
    parser.add_argument("--build-dir", default="rwkv7_state_fwd_fp16_build")
    parser.add_argument("--source-root", default="albatross")
    parser.add_argument("--python-include", default="")
    parser.add_argument("--python-lib-dir", default="")
    args = parser.parse_args()

    output_library = build_kernel(args)
    print(f"Built Albatross kernel: {output_library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
