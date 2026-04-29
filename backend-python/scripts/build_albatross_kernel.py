import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
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
        if arch.startswith("compute"):
            prefix = "compute"
            digits = arch[len(prefix) :]
        elif arch.startswith("sm"):
            prefix = "sm"
            digits = arch[2:]
        elif "." in arch:
            prefix = "sm"
            major, minor = arch.split(".", 1)
            digits = f"{major}{minor}"
        else:
            prefix = "sm"
            digits = arch
        if not digits.isdigit() or len(digits) < 2:
            raise ValueError(f"Invalid CUDA arch: {raw_arch}")
        normalized = f"{prefix}{digits}"
        if normalized not in arches:
            arches.append(normalized)
    if not arches:
        raise ValueError("At least one CUDA arch is required")
    return arches


def resolve_arches(
    raw_arches: str,
    cuda_available=None,
    get_device_capability=None,
) -> list[str]:
    if raw_arches.strip().lower() != "auto":
        return parse_arches(raw_arches)

    cuda_available = cuda_available or torch.cuda.is_available
    get_device_capability = get_device_capability or torch.cuda.get_device_capability
    if not cuda_available():
        raise RuntimeError(
            "Cannot resolve --arch auto because CUDA is not available. "
            "Pass --arch sm80, sm86, sm89, sm90, sm120, or another explicit arch."
        )
    major, minor = get_device_capability()
    return [f"sm{major}{minor}"]


def torch_cuda_arch_list(arches: list[str]) -> str:
    converted = []
    for arch in arches:
        if arch.startswith("compute"):
            digits = arch[len("compute") :]
            converted.append(f"{digits[:-1]}.{digits[-1]}+PTX")
        else:
            digits = arch[2:] if arch.startswith("sm") else arch
            converted.append(f"{digits[:-1]}.{digits[-1]}")
    return ";".join(converted)


def cuda_gencode_flags(arches: list[str]) -> list[str]:
    flags = []
    for arch in arches:
        if arch.startswith("compute"):
            digits = arch[len("compute") :]
            flags.append(f"-gencode=arch=compute_{digits},code=compute_{digits}")
        else:
            digits = arch[2:] if arch.startswith("sm") else arch
            flags.append(f"-gencode=arch=compute_{digits},code=sm_{digits}")
    return flags


def library_suffix() -> str:
    return ".pyd" if os.name == "nt" else ".so"


def arch_directory_name(arches: list[str]) -> str:
    return "_".join(arches)


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


def clear_stale_build_lock(build_dir: Path) -> bool:
    lock_path = build_dir / "lock"
    if not lock_path.exists():
        return False
    lock_path.unlink()
    return True


def ninja_command(max_jobs: str) -> list[str]:
    return ["ninja", "-v", "-j", max_jobs]


def run_ninja_build(build_dir: Path, timeout: int, max_jobs: str) -> None:
    try:
        subprocess.run(
            ninja_command(max_jobs),
            cwd=str(build_dir),
            check=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Albatross kernel ninja build timed out after {timeout}s. "
            f"Build directory: {build_dir}"
        ) from error


def run_command(command: list[str], cwd: Path, timeout: int) -> None:
    try:
        subprocess.run(command, cwd=str(cwd), check=True, timeout=timeout)
    except subprocess.TimeoutExpired as error:
        raise TimeoutError(
            f"Command timed out after {timeout}s: {' '.join(command)}"
        ) from error


def find_python310_dev_paths(env: dict[str, str] | None = None) -> tuple[Path, Path] | None:
    env = env or os.environ
    python_lib_name = "python310.lib"
    candidates = []

    explicit_root = env.get("ALBATROSS_PYTHON310_ROOT")
    if explicit_root:
        candidates.append(Path(explicit_root))

    local_app_data = env.get("LOCALAPPDATA")
    if local_app_data:
        candidates.append(Path(local_app_data) / "Programs" / "Python" / "Python310")

    for key in ("ProgramFiles", "ProgramFiles(x86)"):
        program_files = env.get(key)
        if program_files:
            candidates.append(Path(program_files) / "Python310")
            candidates.append(Path(program_files) / "Python" / "Python310")

    seen = set()
    for root in candidates:
        root_key = str(root).lower()
        if root_key in seen:
            continue
        seen.add(root_key)
        include_dir = root / "include"
        lib_dir = root / "libs"
        if (include_dir / "Python.h").is_file() and (lib_dir / python_lib_name).is_file():
            return include_dir, lib_dir
    return None


def apply_python_dev_paths(args, env: dict[str, str] | None = None) -> None:
    env = env or os.environ
    if args.python_include:
        env["ALBATROSS_PYTHON_INCLUDE"] = args.python_include
    if args.python_lib_dir:
        env["ALBATROSS_PYTHON_LIB_DIR"] = args.python_lib_dir

    if env.get("ALBATROSS_PYTHON_INCLUDE") and env.get("ALBATROSS_PYTHON_LIB_DIR"):
        return

    detected = find_python310_dev_paths(env)
    if detected is None:
        return

    include_dir, lib_dir = detected
    env.setdefault("ALBATROSS_PYTHON_INCLUDE", str(include_dir))
    env.setdefault("ALBATROSS_PYTHON_LIB_DIR", str(lib_dir))


def update_manifest(manifest_path: Path, entry: dict) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    else:
        manifest = {"kernels": []}

    def same_context(candidate: dict) -> bool:
        keys = ["name", "torch", "cuda", "python_abi", "platform", "arch"]
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


def torch_paths() -> tuple[Path, Path, Path]:
    torch_root = Path(torch.__file__).resolve().parent
    torch_include = torch_root / "include"
    torch_api_include = torch_include / "torch" / "csrc" / "api" / "include"
    torch_lib = torch_root / "lib"
    return torch_include, torch_api_include, torch_lib


def cuda_paths() -> tuple[Path, Path, Path]:
    cuda_root = Path(os.environ.get("CUDA_PATH", r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"))
    return cuda_root, cuda_root / "include", cuda_root / "lib" / "x64"


def build_kernel_direct_windows(
    source_root: Path,
    build_dir: Path,
    arches: list[str],
    timeout: int,
) -> Path:
    python_include = Path(os.environ["ALBATROSS_PYTHON_INCLUDE"])
    python_lib_dir = Path(os.environ["ALBATROSS_PYTHON_LIB_DIR"])
    torch_include, torch_api_include, torch_lib = torch_paths()
    _, cuda_include, cuda_lib = cuda_paths()
    cpp_source = source_root / "cuda" / f"{KERNEL_NAME}.cpp"
    cuda_source = source_root / "cuda" / f"{KERNEL_NAME}.cu"
    cpp_object = build_dir / f"{KERNEL_NAME}.o"
    cuda_object = build_dir / f"{KERNEL_NAME}.cuda.o"
    output_library = build_dir / f"{KERNEL_NAME}.pyd"

    include_flags = [
        f"-I{python_include}",
        f"-I{torch_include}",
        f"-I{torch_api_include}",
        f"-I{cuda_include}",
    ]
    common_defines = [
        f"-DTORCH_EXTENSION_NAME={KERNEL_NAME}",
        "-DTORCH_API_INCLUDE_EXTENSION_H",
        "-D_GLIBCXX_USE_CXX11_ABI=0",
    ]

    cl_command = [
        "cl",
        *common_defines,
        *include_flags,
        "/MD",
        "/wd4819",
        "/wd4251",
        "/wd4244",
        "/wd4267",
        "/wd4275",
        "/wd4018",
        "/wd4190",
        "/wd4624",
        "/wd4067",
        "/wd4068",
        "/EHsc",
        "/std:c++17",
        "-c",
        str(cpp_source),
        f"/Fo{cpp_object}",
    ]
    nvcc_command = [
        "nvcc",
        *common_defines,
        *include_flags,
        "-D__CUDA_NO_HALF_OPERATORS__",
        "-D__CUDA_NO_HALF_CONVERSIONS__",
        "-D__CUDA_NO_BFLOAT16_CONVERSIONS__",
        "-D__CUDA_NO_HALF2_OPERATORS__",
        "--expt-relaxed-constexpr",
        *cuda_gencode_flags(arches),
        "-std=c++17",
        "-res-usage",
        "--use_fast_math",
        "-O3",
        "--extra-device-vectorization",
        f"-D_N_={HEAD_SIZE}",
        "-Xcompiler",
        "/EHsc",
        "-Xcompiler",
        "/MD",
        "-c",
        str(cuda_source),
        "-o",
        str(cuda_object),
    ]
    link_command = [
        "link",
        str(cpp_object),
        str(cuda_object),
        "/nologo",
        "/DLL",
        f"/LIBPATH:{python_lib_dir}",
        f"/LIBPATH:{torch_lib}",
        f"/LIBPATH:{cuda_lib}",
        "c10.lib",
        "c10_cuda.lib",
        "torch_cpu.lib",
        "torch_cuda.lib",
        "-INCLUDE:?warp_size@cuda@at@@YAHXZ",
        "torch.lib",
        "torch_python.lib",
        "cudart.lib",
        f"/out:{output_library}",
    ]

    print("Compiling Albatross C++ extension object with cl")
    run_command(cl_command, build_dir, timeout)
    print("Compiling Albatross CUDA extension object with nvcc")
    run_command(nvcc_command, build_dir, timeout)
    print("Linking Albatross extension library")
    run_command(link_command, build_dir, timeout)
    return output_library


def build_kernel(args) -> Path:
    apply_python_dev_paths(args)

    validate_python_extension_build_environment()
    arches = resolve_arches(args.arch)
    os.environ["TORCH_CUDA_ARCH_LIST"] = torch_cuda_arch_list(arches)

    repo_backend = Path(__file__).resolve().parent.parent
    source_root = (repo_backend / args.source_root).resolve()
    output_root = (repo_backend / args.output_root).resolve()
    build_dir = (repo_backend / args.build_dir).resolve()
    build_dir.mkdir(parents=True, exist_ok=True)
    if clear_stale_build_lock(build_dir):
        print(f"Removed stale torch extension build lock: {build_dir / 'lock'}")

    if args.build_system == "direct" and os.name == "nt":
        built_library = build_kernel_direct_windows(
            source_root, build_dir, arches, args.command_timeout
        )
    else:
        from torch.utils.cpp_extension import (
            verify_ninja_availability,
            _write_ninja_file_to_build_library,
        )

        verify_ninja_availability()
        sources = [
            str(source_root / "cuda" / f"{KERNEL_NAME}.cpp"),
            str(source_root / "cuda" / f"{KERNEL_NAME}.cu"),
        ]
        extension_options = extension_build_options()
        _write_ninja_file_to_build_library(
            path=str(build_dir / "build.ninja"),
            name=KERNEL_NAME,
            sources=sources,
            extra_cflags=[],
            extra_cuda_cflags=[
                "-res-usage",
                "--use_fast_math",
                "-O3",
                "--extra-device-vectorization",
                f"-D_N_={HEAD_SIZE}",
            ]
            + (["-Xptxas -O3"] if os.name != "nt" else []),
            extra_sycl_cflags=[],
            extra_ldflags=extension_options["extra_ldflags"],
            extra_include_paths=extension_options["extra_include_paths"],
            with_cuda=True,
            with_sycl=False,
            is_standalone=False,
        )
        print(f"Wrote ninja build file: {build_dir / 'build.ninja'}")
        print(
            f"Running ninja with MAX_JOBS={args.max_jobs}, timeout={args.ninja_timeout}s"
        )
        run_ninja_build(build_dir, args.ninja_timeout, args.max_jobs)
        built_library = find_built_library(build_dir)

    relative_library = (
        Path(f"torch-{torch.__version__}")
        / current_platform_tag()
        / current_python_abi()
        / arch_directory_name(arches)
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
    parser.add_argument("--arch", default="auto")
    parser.add_argument("--output-root", default="albatross/kernels")
    parser.add_argument("--build-dir", default="rwkv7_state_fwd_fp16_build")
    parser.add_argument("--source-root", default="albatross")
    parser.add_argument("--python-include", default="")
    parser.add_argument("--python-lib-dir", default="")
    parser.add_argument("--max-jobs", default=os.environ.get("MAX_JOBS", "1"))
    parser.add_argument("--ninja-timeout", type=int, default=1800)
    parser.add_argument("--command-timeout", type=int, default=1800)
    parser.add_argument(
        "--build-system",
        choices=["direct", "ninja"],
        default="direct" if os.name == "nt" else "ninja",
    )
    args = parser.parse_args()

    output_library = build_kernel(args)
    print(f"Built Albatross kernel: {output_library}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
