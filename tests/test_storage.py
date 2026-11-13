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

    def test_hash_bytes_is_stable_and_content_addressed(self):
        self.assertEqual(storage.hash_bytes(b"abc"), storage.hash_bytes(b"abc"))
        self.assertNotEqual(storage.hash_bytes(b"abc"), storage.hash_bytes(b"abd"))
        # matches the digest embedded in the persisted blob path
        path = storage.persist_blob(b"abc", "text/plain", "f.txt")
        self.assertIn(storage.hash_bytes(b"abc"), path)

    def test_corpus_id_persists_across_calls(self):
        first = storage.load_or_create_corpus_id()
        second = storage.load_or_create_corpus_id()
        self.assertTrue(first)
        self.assertEqual(first, second)

    def test_set_corpus_id_overwrites_persisted_value(self):
        storage.load_or_create_corpus_id()
        storage.set_corpus_id("fixed-corpus")
        self.assertEqual(storage.load_or_create_corpus_id(), "fixed-corpus")


    def test_corpus_blobs_are_separate_per_session(self):
        a, b = storage.new_corpus_id(), storage.new_corpus_id()
        path_a = storage.persist_blob(b"same", "image/png", "x.png", corpus_id=a)
        path_b = storage.persist_blob(b"same", "image/png", "x.png", corpus_id=b)
        self.assertNotEqual(path_a, path_b)
        storage.delete_blob(path_a)
        self.assertEqual(storage.read_blob(path_b), b"same")

    def test_corpus_dir_rejects_path_tricks(self):
        with self.assertRaises(ValueError):
            storage.corpus_dir("../../etc")

    def test_idle_corpora_are_found_and_forgotten(self):
        old, fresh = storage.new_corpus_id(), storage.new_corpus_id()
        storage.touch_corpus(old, now=1000.0)
        storage.touch_corpus(fresh, now=5000.0)
        blob = storage.persist_blob(b"data", "image/png", "x.png", corpus_id=old)
        self.assertEqual(storage.stale_corpora(3600, now=5000.0), [old])
        storage.forget_corpus(old)
        self.assertFalse(Path(blob).exists())
        self.assertEqual(storage.stale_corpora(0, now=5001.0), [fresh])

    def test_touch_is_throttled(self):
        cid = storage.new_corpus_id()
        self.assertTrue(storage.touch_corpus(cid, now=1000.0))
        self.assertFalse(storage.touch_corpus(cid, now=1100.0))
        self.assertTrue(storage.touch_corpus(cid, now=2000.0))

if __name__ == "__main__":
    unittest.main()
