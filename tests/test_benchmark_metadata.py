"""The synthetic metadata benchmark (scripts/benchmark_metadata.py) runs end to end on a small catalog."""
import contextlib
import importlib.util
import io
import json
import logging
from pathlib import Path
import unittest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "benchmark_metadata.py"


class BenchmarkMetadataTests(unittest.TestCase):
    def test_small_run_reports_every_measurement(self):
        spec = importlib.util.spec_from_file_location("benchmark_metadata", SCRIPT)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        # The synthetic index has no mean vector, which the loader reports as a warning.
        logging.disable(logging.WARNING)
        self.addCleanup(logging.disable, logging.NOTSET)
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            module.main(["--items", "200", "--results", "10", "--repeats", "1"])
        report = json.loads(output.getvalue())
        self.assertEqual(report["vector_exact"]["results"], 10)
        self.assertEqual(report["name_substring_all_rows_match"]["results"], 10)


if __name__ == "__main__":
    unittest.main()
