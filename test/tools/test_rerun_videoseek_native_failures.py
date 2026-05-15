import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "rerun_videoseek_native_failures.py"
SPEC = importlib.util.spec_from_file_location("rerun_videoseek_native_failures", MODULE_PATH)
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)


class TestRerunVideoSeekNativeFailures(unittest.TestCase):
    def test_classify_existing_sample_marks_invalid_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sample_dir = Path(tmpdir)
            (sample_dir / "prediction.json").write_text(
                json.dumps({"prediction": "I think the answer is Whitehead."}),
                encoding="utf-8",
            )

            status, detail = tool.classify_existing_sample(sample_dir)

            self.assertEqual(status, "invalid_prediction")
            self.assertIn("Whitehead", detail)

    def test_discover_existing_samples_parses_native_dirs(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            sample_dir = run_dir / "videomme_long_w_subtitle_doc12_idx12"
            sample_dir.mkdir()

            discovered = tool.discover_existing_samples(run_dir)

            self.assertIn(("videomme_long_w_subtitle", 12), discovered)
            self.assertEqual(discovered[("videomme_long_w_subtitle", 12)]["index"], 12)


if __name__ == "__main__":
    unittest.main()
