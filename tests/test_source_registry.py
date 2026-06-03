import tempfile
import unittest
from pathlib import Path

import source_registry


class SourceRegistryTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.original = source_registry.REGISTRY_PATH
        source_registry.REGISTRY_PATH = Path(self.tmp_dir.name) / "source_registry.json"

    def tearDown(self):
        source_registry.REGISTRY_PATH = self.original
        self.tmp_dir.cleanup()

    def test_upsert_and_list(self):
        source_registry.upsert_source({"source_id": "s1", "name": "doc", "corpus_id": "c1"})
        source_registry.upsert_source({"source_id": "s2", "name": "doc2", "corpus_id": "c2"})
        source_registry.upsert_source({"source_id": "s1", "name": "doc-updated", "corpus_id": "c1"})
        rows = source_registry.list_sources("c1")
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["name"], "doc-updated")

    def test_delete_source(self):
        source_registry.upsert_source({"source_id": "s1", "name": "doc", "corpus_id": "c1"})
        source_registry.delete_source("s1")
        self.assertEqual(source_registry.list_sources("c1"), [])

    def test_atomic_write_leaves_no_temp_files(self):
        source_registry.upsert_source({"source_id": "s1", "name": "doc", "corpus_id": "c1"})
        source_registry.upsert_source({"source_id": "s2", "name": "doc2", "corpus_id": "c1"})
        leftover = list(Path(self.tmp_dir.name).glob("*.tmp"))
        self.assertEqual(leftover, [])
        self.assertTrue(source_registry.REGISTRY_PATH.exists())
        # File is complete, valid JSON after the replace.
        self.assertEqual(len(source_registry.list_sources("c1")), 2)


if __name__ == "__main__":
    unittest.main()
