import json
import tempfile
import unittest
from pathlib import Path

from albatross.kernel_loader import RuntimeKernelContext, find_precompiled_kernel


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


if __name__ == "__main__":
    unittest.main()
