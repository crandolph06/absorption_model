import tempfile
import unittest
from pathlib import Path

import yaml

from src.viability.config import load_config


class ViabilityConfigTest(unittest.TestCase):
    def test_example_config_loads(self):
        config = load_config("configs/viability.example.yaml")
        self.assertEqual(config.run.name, "viability_smoke")
        self.assertEqual(config.model.expected_brain_outputs, 16)

    def test_empty_config_file_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "empty.yaml"
            path.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "empty"):
                load_config(path)

    def test_missing_top_level_section_fails(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "partial.yaml"
            path.write_text(yaml.safe_dump({"run": {"name": "x"}}), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required config key: model"):
                load_config(path)

    def test_missing_nested_field_fails_with_dotted_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "missing_field.yaml"
            data = yaml.safe_load(Path("configs/viability.example.yaml").read_text(encoding="utf-8"))
            del data["model"]["brain_path"]
            path.write_text(yaml.safe_dump(data), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "Missing required config key: model.brain_path"):
                load_config(path)

    def test_dump_resolved_config_round_trip(self):
        config = load_config("configs/viability.example.yaml")
        with tempfile.TemporaryDirectory() as tmp:
            resolved = config.dump_resolved_config(Path(tmp) / "config_resolved.yaml")
            reloaded = load_config(resolved)
            self.assertEqual(config.to_dict(), reloaded.to_dict())


if __name__ == "__main__":
    unittest.main()
