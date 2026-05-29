import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).resolve().parents[2] / "tools" / "evaluate_videoseek_videomme_v2_runs.py"
SPEC = importlib.util.spec_from_file_location("evaluate_videoseek_videomme_v2_runs", MODULE_PATH)
tool = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(tool)


class TestEvaluateVideoSeekVideoMMEV2Runs(unittest.TestCase):
    def test_parse_sample_dir_name(self):
        parsed = tool.parse_sample_dir_name(Path("videomme_v2_doc55_idx61"))
        self.assertEqual(parsed["task"], "videomme_v2")
        self.assertEqual(parsed["doc_id"], 55)
        self.assertEqual(parsed["index"], 61)

    def test_discover_samples_prefers_run_summary(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir)
            sample_dir = run_dir / "videomme_v2_doc0_idx0"
            sample_dir.mkdir()
            (run_dir / "run_summary.json").write_text(
                json.dumps(
                    {
                        "samples": [
                            {
                                "task": "videomme_v2",
                                "doc_id": 0,
                                "index": 0,
                                "sample_dir": str(sample_dir),
                            }
                        ]
                    }
                ),
                encoding="utf-8",
            )

            samples = tool.discover_samples(run_dir)
            self.assertEqual(len(samples), 1)
            self.assertEqual(samples[0]["sample_dir"], str(sample_dir))


if __name__ == "__main__":
    unittest.main()
