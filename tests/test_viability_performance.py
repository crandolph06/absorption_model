from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.viability.config import load_config
from src.viability.performance import (
    benchmark_evaluation_batch,
    parse_worker_counts,
    prepare_benchmark_config,
)


class ViabilityPerformanceTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_parse_worker_counts_preserves_order_and_dedupes(self):
        self.assertEqual(parse_worker_counts("1, 4, 4,8"), [1, 4, 8])

    def test_parse_worker_counts_rejects_empty_or_nonpositive(self):
        with self.assertRaisesRegex(ValueError, "At least one"):
            parse_worker_counts(" , ")
        with self.assertRaisesRegex(ValueError, "positive"):
            parse_worker_counts("1,0")

    def test_prepare_benchmark_config_can_force_one_year_physics(self):
        prepared = prepare_benchmark_config(
            self.config,
            phase_backend="physics",
            years_to_run=1,
        )

        self.assertEqual(prepared.model.phase_backend, "physics")
        self.assertIsNone(prepared.model.brain_path)
        self.assertIsNone(prepared.model.expected_brain_outputs)
        self.assertEqual(prepared.model.years_to_run, 1)
        self.assertEqual(prepared.model.assessment_start_year, self.config.model.start_year)
        self.assertEqual(prepared.model.target_year, self.config.model.start_year)

    def test_prepare_benchmark_config_rejects_invalid_years(self):
        with self.assertRaisesRegex(ValueError, "years_to_run"):
            prepare_benchmark_config(self.config, years_to_run=0)

    def test_benchmark_evaluation_batch_writes_summary_without_running_physics(self):
        config = prepare_benchmark_config(
            replace(self.config, run=replace(self.config.run, workers=1)),
            phase_backend="physics",
            years_to_run=1,
        )

        def fake_evaluate(designs, _config, workers=None, checkpoint_dir=None, checkpoint_every=100):
            self.assertIn(workers, {1, 2})
            self.assertIsNotNone(checkpoint_dir)
            self.assertEqual(checkpoint_every, 7)
            return pd.DataFrame(
                {
                    "design_id": designs["design_id"],
                    "status": ["ok"] * len(designs),
                    "phi": [-1.0] * len(designs),
                }
            )

        with tempfile.TemporaryDirectory() as tmp:
            with patch("src.viability.performance.evaluate_designs_parallel", side_effect=fake_evaluate):
                designs, summaries = benchmark_evaluation_batch(
                    config=config,
                    output_dir=tmp,
                    n=3,
                    worker_counts=[1, 2],
                    method="random",
                    checkpoint_every=7,
                )

            self.assertEqual(len(designs), 3)
            self.assertEqual([summary.workers for summary in summaries], [1, 2])
            self.assertTrue((Path(tmp) / "benchmark_summary.csv").exists())
            self.assertTrue((Path(tmp) / "benchmark_summary.json").exists())
            self.assertTrue((Path(tmp) / "workers_01" / "evaluations.parquet").exists())
            self.assertTrue((Path(tmp) / "workers_02" / "evaluations.parquet").exists())


if __name__ == "__main__":
    unittest.main()
