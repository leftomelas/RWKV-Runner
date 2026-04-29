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

    def test_resolve_arches_auto_uses_visible_cuda_device(self):
        module = load_build_script()

        self.assertEqual(
            module.resolve_arches(
                "auto",
                cuda_available=lambda: True,
                get_device_capability=lambda: (12, 0),
            ),
            ["sm120"],
        )

    def test_torch_cuda_arch_list_converts_to_decimal_form(self):
        module = load_build_script()

        self.assertEqual(
            module.torch_cuda_arch_list(["sm80", "sm86", "sm120"]),
            "8.0;8.6;12.0",
        )

    def test_cuda_gencode_flags_uses_sm_targets(self):
        module = load_build_script()

        self.assertEqual(
            module.cuda_gencode_flags(["sm120"]),
            ["-gencode=arch=compute_120,code=sm_120"],
        )

    def test_find_python310_dev_paths_uses_default_localappdata_layout(self):
        module = load_build_script()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            python_root = root / "Programs" / "Python" / "Python310"
            include_dir = python_root / "include"
            lib_dir = python_root / "libs"
            include_dir.mkdir(parents=True)
            lib_dir.mkdir()
            (include_dir / "Python.h").write_text("", encoding="utf-8")
            (lib_dir / "python310.lib").write_text("", encoding="utf-8")

            result = module.find_python310_dev_paths(
                {
                    "LOCALAPPDATA": str(root),
                    "ProgramFiles": str(root / "ProgramFiles"),
                    "ProgramFiles(x86)": str(root / "ProgramFilesX86"),
                }
            )

            self.assertEqual(result, (include_dir, lib_dir))

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

    def test_clear_stale_build_lock_removes_empty_lock(self):
        module = load_build_script()
        with tempfile.TemporaryDirectory() as tmp:
            build_dir = Path(tmp)
            lock_path = build_dir / "lock"
            lock_path.write_text("", encoding="utf-8")

            self.assertTrue(module.clear_stale_build_lock(build_dir))
            self.assertFalse(lock_path.exists())

    def test_ninja_command_uses_single_job_by_default(self):
        module = load_build_script()

        self.assertEqual(module.ninja_command("1"), ["ninja", "-v", "-j", "1"])

    def test_vs2022_wrapper_script_exists(self):
        script_path = (
            Path(__file__).resolve().parent.parent
            / "scripts"
            / "build_albatross_kernel_vs2022.cmd"
        )

        self.assertTrue(script_path.is_file())
        script = script_path.read_text(encoding="utf-8")
        self.assertIn("vcvars64.bat", script)
        self.assertIn("build_albatross_kernel.py", script)
        self.assertIn("MAX_JOBS", script)


if __name__ == "__main__":
    unittest.main()
