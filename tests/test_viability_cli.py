import contextlib
import io
import json
import tempfile
import unittest
from unittest.mock import patch

import pandas as pd

from src.viability import cli


class ViabilityCliTest(unittest.TestCase):
    def test_run_doe_reports_and_passes_worker_override(self):
        designs = pd.DataFrame({"design_id": ["d0"]})
        evaluations = pd.DataFrame({"design_id": ["d0"], "status": ["ok"], "phi": [-1.0]})

        with tempfile.TemporaryDirectory() as tmp:
            argv = [
                "viability",
                "run-doe",
                "--config",
                "configs/viability.example.yaml",
                "--n",
                "1",
                "--workers",
                "6",
                "--output-dir",
                tmp,
            ]
            with (
                patch("sys.argv", argv),
                patch("src.viability.cli.generate_doe", return_value=designs),
                patch("src.viability.cli.evaluate_designs_parallel", return_value=evaluations) as evaluate,
                contextlib.redirect_stdout(io.StringIO()) as stdout,
            ):
                status = cli.main()

        self.assertEqual(status, 0)
        self.assertEqual(evaluate.call_args.kwargs["workers"], 6)
        payload = json.loads(stdout.getvalue())
        self.assertEqual(payload["workers"], 6)
        self.assertEqual(payload["design_count"], 1)
        self.assertEqual(payload["evaluated_count"], 1)


if __name__ == "__main__":
    unittest.main()
