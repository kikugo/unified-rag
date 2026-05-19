import tempfile
import unittest
from pathlib import Path

import storage


class StorageTest(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.original_root = storage.STORAGE_ROOT
        storage.STORAGE_ROOT = Path(self.tmp_dir.name) / "blobstore"

    def tearDown(self):
        storage.STORAGE_ROOT = self.original_root
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


if __name__ == "__main__":
    unittest.main()
