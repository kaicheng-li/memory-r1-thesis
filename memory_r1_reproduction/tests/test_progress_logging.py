"""Logging must remain visible before long calls and on file-write failure."""
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout, redirect_stderr
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from memfactory.progress import append_progress


class ProgressLoggingTests(unittest.TestCase):
    def test_creates_parent_and_emits_same_record_to_console_and_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "manager" / "training_progress.jsonl"
            console = io.StringIO()
            with redirect_stdout(console):
                append_progress(path, "model_load_start", component="manager")
            self.assertEqual(console.getvalue(), path.read_text(encoding="utf-8"))
            self.assertEqual(json.loads(console.getvalue())["event"], "model_load_start")

    def test_file_failure_is_visible_and_does_not_abort_training(self):
        with tempfile.TemporaryDirectory() as directory:
            console, errors = io.StringIO(), io.StringIO()
            with redirect_stdout(console), redirect_stderr(errors):
                append_progress(directory, "validation_start", epoch=0)
            self.assertEqual(json.loads(console.getvalue())["event"], "validation_start")
            self.assertIn("Cannot write", errors.getvalue())
            self.assertIn(str(Path(directory).resolve()), errors.getvalue())


if __name__ == "__main__":
    unittest.main()
