import ast
import pathlib
import re
import unittest


BACKEND_ROOT = pathlib.Path(__file__).resolve().parents[1]
ALBATROSS_CONSOLE_FILES = [
    BACKEND_ROOT / "albatross" / "rwkv7.py",
    BACKEND_ROOT / "albatross_engine" / "core.py",
    BACKEND_ROOT / "albatross_engine" / "worker.py",
]


def user_visible_message_texts(path: pathlib.Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if isinstance(node.func, ast.Name):
            call_name = node.func.id
        else:
            call_name = ""
        if call_name not in {"print", "RuntimeError"}:
            continue
        if not node.args:
            continue
        first = node.args[0]
        if isinstance(first, ast.Constant) and isinstance(first.value, str):
            yield first.value
        elif isinstance(first, ast.JoinedStr):
            parts = [
                value.value
                for value in first.values
                if isinstance(value, ast.Constant) and isinstance(value.value, str)
            ]
            yield "".join(parts)


class AlbatrossConsoleMessageTests(unittest.TestCase):
    def test_startup_console_messages_are_english(self):
        cjk = re.compile(r"[\u3400-\u9fff]")
        messages = [
            (path.name, message)
            for path in ALBATROSS_CONSOLE_FILES
            for message in user_visible_message_texts(path)
        ]

        self.assertGreater(len(messages), 0)
        self.assertEqual(
            [
                (filename, message)
                for filename, message in messages
                if cjk.search(message)
            ],
            [],
        )

    def test_model_initialization_message_does_not_claim_kernel_compilation(self):
        worker_messages = list(user_visible_message_texts(BACKEND_ROOT / "albatross_engine" / "worker.py"))

        self.assertIn("[] Loading and initializing model...", worker_messages)
        self.assertFalse(
            any("compile" in message.lower() and "model" in message.lower() for message in worker_messages)
        )


if __name__ == "__main__":
    unittest.main()
