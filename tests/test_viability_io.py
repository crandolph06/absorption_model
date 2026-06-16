import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd

from src.viability.config import load_config
from src.viability.io import run_output_dir, write_config_resolved, write_table


class ViabilityIoTest(unittest.TestCase):
    def setUp(self):
        self.config = load_config("configs/viability.example.yaml")

    def test_run_output_dir(self):
        path = run_output_dir(self.config)
        self.assertEqual(path, Path("outputs/viability/runs/viability_smoke"))

    def test_write_config_resolved(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = write_config_resolved(self.config, tmp)
            self.assertTrue(path.exists())
            self.assertEqual(path.name, "config_resolved.yaml")

    def test_write_table_falls_back_to_csv(self):
        df = pd.DataFrame({"a": [1, 2]})
        with tempfile.TemporaryDirectory() as tmp:
            with patch("pandas.DataFrame.to_parquet", side_effect=RuntimeError("no parquet")):
                path = write_table(df, Path(tmp) / "table.parquet")
            self.assertEqual(path.suffix, ".csv")
            self.assertTrue(path.exists())


if __name__ == "__main__":
    unittest.main()
