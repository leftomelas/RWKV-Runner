import json
import os
import tempfile
import unittest
from pathlib import Path

from albatross.kernel_loader import (
    RuntimeKernelContext,
    extension_build_options,
    find_precompiled_kernel,
    validate_python_extension_build_environment,
)


class AlbatrossKernelLoaderTests(unittest.TestCase):
    def test_find_precompiled_kernel_matches_runtime_context(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            library = (
                root
                / "torch-2.7.1+cu128"
                / "win_amd64"
                / "cp310"
                / "rwkv7_state_fwd_fp16.pyd"
            )
            library.parent.mkdir(parents=True)
            library.write_bytes(b"fake")
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kernels": [
                            {
                                "name": "rwkv7_state_fwd_fp16",
                                "torch": "2.7.1+cu128",
                                "cuda": "12.8",
                                "python_abi": "cp310",
                                "platform": "win_amd64",
                                "arch": ["sm80", "sm86", "sm89"],
                                "path": "torch-2.7.1+cu128/win_amd64/cp310/rwkv7_state_fwd_fp16.pyd",
                                "source": "local-build",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = find_precompiled_kernel(
                manifest,
                RuntimeKernelContext(
                    torch_version="2.7.1+cu128",
                    cuda_version="12.8",
                    python_abi="cp310",
                    platform_tag="win_amd64",
                    cuda_arch="sm89",
                ),
            )

            self.assertEqual(result, library)

    def test_find_precompiled_kernel_returns_none_for_arch_mismatch(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest = root / "manifest.json"
            manifest.write_text(
                json.dumps(
                    {
                        "kernels": [
                            {
                                "name": "rwkv7_state_fwd_fp16",
                                "torch": "2.7.1+cu128",
                                "cuda": "12.8",
                                "python_abi": "cp310",
                                "platform": "win_amd64",
                                "arch": ["sm80"],
                                "path": "missing.pyd",
                                "source": "local-build",
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            result = find_precompiled_kernel(
                manifest,
                RuntimeKernelContext(
                    torch_version="2.7.1+cu128",
                    cuda_version="12.8",
                    python_abi="cp310",
                    platform_tag="win_amd64",
                    cuda_arch="sm89",
                ),
            )

            self.assertIsNone(result)

    def test_extension_build_options_uses_python_include_and_lib_env(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            include_dir = root / "include"
            lib_dir = root / "libs"
            include_dir.mkdir()
            lib_dir.mkdir()

            options = extension_build_options(
                {
                    "ALBATROSS_PYTHON_INCLUDE": str(include_dir),
                    "ALBATROSS_PYTHON_LIB_DIR": str(lib_dir),
                }
            )

            self.assertIn(str(include_dir), options["extra_include_paths"])
            if os.name == "nt":
                self.assertIn(f"/LIBPATH:{lib_dir}", options["extra_ldflags"])
            else:
                self.assertIn(f"-L{lib_dir}", options["extra_ldflags"])

    def test_validate_python_extension_build_environment_accepts_env_header(self):
        with tempfile.TemporaryDirectory() as tmp:
            include_dir = Path(tmp) / "include"
            include_dir.mkdir()
            (include_dir / "Python.h").write_text("", encoding="utf-8")

            validate_python_extension_build_environment(
                {"ALBATROSS_PYTHON_INCLUDE": str(include_dir)}
            )

    def test_validate_python_extension_build_environment_reports_missing_header(self):
        with self.assertRaisesRegex(RuntimeError, "Python.h"):
            validate_python_extension_build_environment(
                {},
                python_include_dir=Path(tempfile.gettempdir())
                / "missing-albatross-python-include",
            )


if __name__ == "__main__":
    unittest.main()
