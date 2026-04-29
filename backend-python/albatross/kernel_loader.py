from dataclasses import dataclass
import json
from pathlib import Path
import platform
import sys

import torch


@dataclass(frozen=True)
class RuntimeKernelContext:
    torch_version: str
    cuda_version: str | None
    python_abi: str
    platform_tag: str
    cuda_arch: str


def find_precompiled_kernel(
    manifest_path: Path, context: RuntimeKernelContext
) -> Path | None:
    if not manifest_path.is_file():
        return None

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
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
        if context.cuda_arch not in entry.get("arch", []):
            continue
        candidate = manifest_path.parent / entry["path"]
        if candidate.is_file():
            return candidate
    return None


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
    kernel_path = find_precompiled_kernel(manifest_path, current_runtime_context())
    if kernel_path is None:
        return False
    torch.ops.load_library(str(kernel_path))
    return True
