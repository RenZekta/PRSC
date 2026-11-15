import json
import unittest
from pathlib import Path

from prsc.validate import ValidationError, validate_gram


ROOT = Path(__file__).resolve().parents[1]


class KnowledgeGramValidationTests(unittest.TestCase):
    def setUp(self):
        self.example = json.loads((ROOT / "examples" / "knowledge-gram.example.json").read_text())

    def test_example_is_valid(self):
        validate_gram(self.example)

    def test_missing_field_is_rejected(self):
        broken = dict(self.example)
        broken.pop("gram_id")
        with self.assertRaises(ValidationError):
            validate_gram(broken)

    def test_confidence_out_of_range_is_rejected(self):
        broken = json.loads(json.dumps(self.example))
        broken["residual"]["confidence"] = 1.5
        with self.assertRaises(ValidationError):
            validate_gram(broken)

    def test_unknown_activation_mode_is_rejected(self):
        broken = json.loads(json.dumps(self.example))
        broken["activation"]["preferred_mode"] = "magic_pointer"
        with self.assertRaises(ValidationError):
            validate_gram(broken)


if __name__ == "__main__":
    unittest.main()
