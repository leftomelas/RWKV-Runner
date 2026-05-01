from dataclasses import dataclass
import json
import os
from pathlib import Path
import platform
import sys
import sysconfig

import torch


DEFAULT_PREFERRED_KERNEL_ARCH = "sm80_compute80"


@dataclass(frozen=True)
class RuntimeKernelContext:
    torch_version: str
    cuda_version: str | None
    python_abi: str
    platform_tag: str
    cuda_arch: str


def _arch_digits(arch: str, prefix: str) -> int | None:
    if not arch.startswith(prefix):
        return None
    digits = arch[len(prefix) :]
    if not digits.isdigit():
        return None
    return int(digits)


def _supports_runtime_arch(entry_arches: list[str], runtime_arch: str) -> bool:
    if runtime_arch in entry_arches:
        return True
    runtime_digits = _arch_digits(runtime_arch, "sm")
    if runtime_digits is None:
        return False
    for entry_arch in entry_arches:
        ptx_digits = _arch_digits(entry_arch, "compute")
        if ptx_digits is not None and ptx_digits <= runtime_digits:
            return True
    return False


def _arch_key(entry_arches: list[str]) -> str:
    return "_".join(entry_arches)


def _normalize_forced_arch(forced_arch: str | None) -> str | None:
    if not forced_arch:
        return None
    return forced_arch.strip().lower().replace(";", "_").replace(",", "_")


def _arch_preference_rank(entry_arches: list[str], preferred_arch: str | None) -> int:
    if not preferred_arch:
        return 1
    return 0 if _arch_key(entry_arches) == _normalize_forced_arch(preferred_arch) else 1


def _arch_rank(entry_arches: list[str], runtime_arch: str) -> tuple[int, int]:
    if runtime_arch in entry_arches:
        return (0, 0)

    best_ptx = -1
    runtime_digits = _arch_digits(runtime_arch, "sm")
    if runtime_digits is None:
        return (2, 0)

    for entry_arch in entry_arches:
        ptx_digits = _arch_digits(entry_arch, "compute")
        if ptx_digits is not None and ptx_digits <= runtime_digits:
            best_ptx = max(best_ptx, ptx_digits)

    if best_ptx >= 0:
        return (1, -best_ptx)
    return (2, 0)


def find_precompiled_kernel(
    manifest_path: Path,
    context: RuntimeKernelContext,
    forced_arch: str | None = None,
    preferred_arch: str | None = None,
) -> Path | None:
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    forced_arch_key = _normalize_forced_arch(forced_arch)
    candidates = []
    for entry in manifest.get("kernels", []):
        if entry.get("name") != "rwkv7_state_fwd_fp16":
            continue
        if entry.get("torch") != context.torch_version:
            continue
        if entry.get("cuda") != context.cuda_version:
            continue
        if entry.get("python_abi") != context.python_abi:
            continue
        if entry.get("platform") != context.platform_tag:
            continue
        entry_arches = entry.get("arch", [])
        if forced_arch_key and _arch_key(entry_arches) != forced_arch_key:
            continue
        if not _supports_runtime_arch(entry_arches, context.cuda_arch):
            continue
        candidate = manifest_path.parent / entry["path"]
        if candidate.is_file():
            candidates.append((
                (
                    _arch_preference_rank(
                        entry_arches,
                        None if forced_arch_key else preferred_arch,
                    ),
                    *_arch_rank(entry_arches, context.cuda_arch),
                ),
                candidate,
            ))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0])
    return candidates[0][1]


def current_python_abi() -> str:
    return f"cp{sys.version_info.major}{sys.version_info.minor}"


def current_platform_tag() -> str:
    system = platform.system().lower()
    machine = platform.machine().lower()
    if system == "windows" and machine in {"amd64", "x86_64"}:
        return "win_amd64"
    if system == "linux" and machine in {"amd64", "x86_64"}:
        return "linux_x86_64"
    return f"{system}_{machine}"


def current_cuda_arch() -> str:
    major, minor = torch.cuda.get_device_capability()
    return f"sm{major}{minor}"


def current_runtime_context() -> RuntimeKernelContext:
    return RuntimeKernelContext(
        torch_version=torch.__version__,
        cuda_version=torch.version.cuda,
        python_abi=current_python_abi(),
        platform_tag=current_platform_tag(),
        cuda_arch=current_cuda_arch(),
    )


def load_precompiled_kernel_if_available(manifest_path: Path) -> bool:
    if not torch.cuda.is_available():
        return False
    kernel_path = find_precompiled_kernel(
        manifest_path,
        current_runtime_context(),
        forced_arch=os.environ.get("ALBATROSS_KERNEL_ARCH"),
        preferred_arch=DEFAULT_PREFERRED_KERNEL_ARCH,
    )
    if kernel_path is None:
        return False
    torch.ops.load_library(str(kernel_path))
    return True


def _split_env_paths(value: str | None) -> list[str]:
    if not value:
        return []
    return [path for path in value.split(os.pathsep) if path]


def extension_build_options(env: dict[str, str] | None = None) -> dict[str, list[str]]:
    env = env or os.environ
    extra_include_paths = _split_env_paths(env.get("ALBATROSS_PYTHON_INCLUDE"))
    extra_include_paths.extend(
        _split_env_paths(env.get("ALBATROSS_TORCH_EXTENSION_INCLUDE_PATHS"))
    )

    extra_ldflags = _split_env_paths(env.get("ALBATROSS_TORCH_EXTENSION_LDFLAGS"))
    for lib_dir in _split_env_paths(env.get("ALBATROSS_PYTHON_LIB_DIR")):
        if os.name == "nt":
            extra_ldflags.append(f"/LIBPATH:{lib_dir}")
        else:
            extra_ldflags.append(f"-L{lib_dir}")

    return {
        "extra_include_paths": extra_include_paths,
        "extra_ldflags": extra_ldflags,
    }


def validate_python_extension_build_environment(
    env: dict[str, str] | None = None,
    python_include_dir: Path | None = None,
    python_lib_dir: Path | None = None,
) -> None:
    env = env or os.environ
    python_include_dir = python_include_dir or Path(sysconfig.get_path("include"))
    include_dirs = [python_include_dir]
    include_dirs.extend(Path(path) for path in _split_env_paths(env.get("ALBATROSS_PYTHON_INCLUDE")))
    include_dirs.extend(
        Path(path)
        for path in _split_env_paths(env.get("ALBATROSS_TORCH_EXTENSION_INCLUDE_PATHS"))
    )

    if any((include_dir / "Python.h").is_file() for include_dir in include_dirs):
        if os.name != "nt":
            return

        python_lib_name = f"python{sys.version_info.major}{sys.version_info.minor}.lib"
        lib_dirs = []
        if python_lib_dir is not None:
            lib_dirs.append(python_lib_dir)
        libdir = sysconfig.get_config_var("LIBDIR")
        if libdir:
            lib_dirs.append(Path(libdir))
        lib_dirs.extend(
            [
                Path(sys.prefix) / "libs",
                Path(sys.base_prefix) / "libs",
            ]
        )
        lib_dirs.extend(
            Path(path) for path in _split_env_paths(env.get("ALBATROSS_PYTHON_LIB_DIR"))
        )
        if any((lib_dir / python_lib_name).is_file() for lib_dir in lib_dirs):
            return
        checked_libs = ", ".join(str(path) for path in lib_dirs)
        raise RuntimeError(
            "Cannot build Albatross CUDA extension because "
            f"{python_lib_name} was not found. Checked: {checked_libs}. "
            "Install the matching full Python distribution, then either copy its "
            "libs directory into py310 or set ALBATROSS_PYTHON_LIB_DIR before "
            "starting the backend."
        )

    checked = ", ".join(str(path) for path in include_dirs)
    raise RuntimeError(
        "Cannot build Albatross CUDA extension because Python.h was not found. "
        f"Checked: {checked}. "
        "RWKV Runner's bundled py310 is an embeddable Python and may not include "
        "the CPython headers required by torch.utils.cpp_extension. Install the "
        "matching full Python 3.10.x distribution, then either copy its Include "
        "and libs directories into py310 or set ALBATROSS_PYTHON_INCLUDE and "
        "ALBATROSS_PYTHON_LIB_DIR before starting the backend."
    )
