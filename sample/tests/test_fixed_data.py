import hashlib
import json
import unittest
from pathlib import Path

from tiny_gpt.tokenizer import DATA_DIR


class FixedDataTests(unittest.TestCase):
    def test_fixed_documents_and_validation_separation(self) -> None:
        expected = json.loads(
            (Path(__file__).parent.parent / "dataset-manifest.json").read_text()
        )
        ids_by_split: dict[str, set[str]] = {}
        texts_by_split: dict[str, set[str]] = {}
        for split in ("train", "validation"):
            path = DATA_DIR / (split + ".jsonl")
            content = path.read_bytes()
            self.assertEqual(
                hashlib.sha256(content).hexdigest(),
                expected["splits"][split]["sha256"],
            )
            ids: set[str] = set()
            texts: set[str] = set()
            for line in content.decode("utf-8").splitlines():
                document = json.loads(line)
                ids.add(document["id"])
                texts.add(document["text"])
            self.assertEqual(len(ids), expected["splits"][split]["documents"])
            self.assertEqual(len(texts), len(ids))
            ids_by_split[split] = ids
            texts_by_split[split] = texts
        self.assertTrue(ids_by_split["train"].isdisjoint(ids_by_split["validation"]))
        self.assertTrue(texts_by_split["train"].isdisjoint(texts_by_split["validation"]))


if __name__ == "__main__":
    unittest.main()
