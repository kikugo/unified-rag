import tempfile
import unittest
from pathlib import Path

import storage


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.original_root = storage.STORAGE_ROOT
        self.original_corpus_path = storage.CORPUS_ID_PATH
        storage.STORAGE_ROOT = Path(self.tmp_dir.name) / "blobstore"
        storage.CORPUS_ID_PATH = storage.STORAGE_ROOT / "corpus_id"

    def tearDown(self):
        storage.STORAGE_ROOT = self.original_root
        storage.CORPUS_ID_PATH = self.original_corpus_path
        self.tmp_dir.cleanup()

    def test_persist_blob_is_deterministic_and_readable(self):
        data = b"hello-world"
        path1 = storage.persist_blob(data, "text/plain", "note.txt")
        path2 = storage.persist_blob(data, "text/plain", "note.txt")

        self.assertEqual(path1, path2)
        self.assertEqual(storage.read_blob(path1), data)

    def test_delete_blob_removes_file(self):
        data = b"to-delete"
        path = storage.persist_blob(data, "application/octet-stream", "chunk.bin")
        self.assertTrue(Path(path).exists())

        storage.delete_blob(path)

        self.assertFalse(Path(path).exists())
        self.assertEqual(storage.read_blob(path), b"")

    def test_corpus_id_persists_across_calls(self):
        first = storage.load_or_create_corpus_id()
        second = storage.load_or_create_corpus_id()
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_set_corpus_id_overwrites_persisted_value(self):
        storage.load_or_create_corpus_id()
        storage.set_corpus_id("fixed-corpus")
        self.assertEqual(storage.load_or_create_corpus_id(), "fixed-corpus")


if __name__ == "__main__":
    unittest.main()
