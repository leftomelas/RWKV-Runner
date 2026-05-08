import json
import os
import subprocess
import sys
import textwrap
import unittest


class AlbatrossLazyImportTests(unittest.TestCase):
    def test_main_import_does_not_import_albatross_engine_modules(self):
        backend_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        script = textwrap.dedent(
            f"""
            import json
            import os
            import sys

            sys.path.insert(0, {backend_root!r})
            os.environ.setdefault("RWKV_RUNNER_PARAMS", "")
            import main  # noqa: F401

            loaded = sorted(
                name for name in sys.modules if name.startswith("albatross_engine")
            )
            print(json.dumps(loaded))
            """
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            check=False,
            capture_output=True,
            text=True,
        )

        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(json.loads(result.stdout), [])


if __name__ == "__main__":
    unittest.main()
