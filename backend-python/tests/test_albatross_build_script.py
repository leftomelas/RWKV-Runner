import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


def load_build_script():
    script_path = (
        Path(__file__).resolve().parent.parent
        / "scripts"
        / "build_albatross_kernel.py"
    )
    spec = importlib.util.spec_from_file_location("build_albatross_kernel", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class AlbatrossBuildScriptTests(unittest.TestCase):
    def test_parse_arches_accepts_sm_and_decimal_forms(self):
        module = load_build_script()

        self.assertEqual(
            module.parse_arches("sm80,8.6,sm120"),
            ["sm80", "sm86", "sm120"],
        )

    def test_torch_cuda_arch_list_converts_to_decimal_form(self):
        module = load_build_script()

        self.assertEqual(
            module.torch_cuda_arch_list(["sm80", "sm86", "sm120"]),
            "8.0;8.6;12.0",
        )

    def test_update_manifest_replaces_matching_context(self):
        module = load_build_script()
        with tempfile.TemporaryDirectory() as tmp:
            manifest_path = Path(tmp) / "manifest.json"
            existing = {
                "name": "rwkv7_state_fwd_fp16",
                "torch": "2.7.1+cu128",
                "cuda": "12.8",
                "python_abi": "cp310",
                "platform": "win_amd64",
                "arch": ["sm80"],
                "path": "old.pyd",
                "source": "local-build",
            }
            replacement = dict(existing)
            replacement["arch"] = ["sm120"]
            replacement["path"] = "new.pyd"
            manifest_path.write_text(
                json.dumps({"kernels": [existing]}),
                encoding="utf-8",
            )

            module.update_manifest(manifest_path, replacement)

            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["kernels"], [replacement])


if __name__ == "__main__":
    unittest.main()
