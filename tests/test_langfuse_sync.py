"""Golden-set sync and traced eval runs against fake Langfuse objects."""
import os
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from test_run import ROWS, _Base

from eval import langfuse_sync
from eval.langfuse_sync import DATASET, dataset_items, item_id, sync
from eval.run import RunConfig, load_prices, main, run, run_traced
from rag import tracing
from rag.cache import DiskCache


class FakeClient:
    def __init__(self):
        self.datasets, self.items, self.flushed = [], [], False

    def create_dataset(self, **kwargs):
        self.datasets.append(kwargs)

    def create_dataset_item(self, **kwargs):
        self.items.append(kwargs)

    def flush(self):
        self.flushed = True


class FakeDataset:
    """Calls the task and evaluators per item, as Langfuse's run_experiment does."""

    def __init__(self, items):
        self.items = items
        self.calls = []

    def run_experiment(self, *, name, run_name, task, evaluators, max_concurrency, **kwargs):
        self.calls.append({"name": name, "run_name": run_name, "max_concurrency": max_concurrency})
        self.scores = {}
        for item in self.items:
            output = task(item=item)
            self.scores[item.id] = [e for ev in evaluators for e in ev(input=item.input, output=output,
                                                                            metadata=item.metadata)]
        return SimpleNamespace(experiment_url="https://example.invalid/run")


def _without_latency(rows):
    return [{k: v for k, v in r.items() if k != "latency_ms"} for r in rows]


def _items(rows):
    return [SimpleNamespace(**item) for item in dataset_items(rows)]


class SyncTest(unittest.TestCase):
    def test_items_have_stable_prefixed_ids_and_carry_the_gold(self):
        items = dataset_items(ROWS)
        self.assertEqual([i["id"] for i in items], [f"{DATASET}-g001", f"{DATASET}-g002", f"{DATASET}-g003"])
        self.assertEqual(items[0]["input"], {"question": ROWS[0]["question"]})
        self.assertEqual(items[0]["expected_output"]["gold_chunk_ids"], ["art5"])
        self.assertEqual(items[2]["metadata"], {"golden_id": "g003", "type": "unanswerable", "answerable": False})
        self.assertEqual(dataset_items(ROWS), items, "same rows, same items")

    def test_sync_creates_dataset_and_upserts_items(self):
        client = FakeClient()
        self.assertEqual(sync(client, ROWS, "abc"), 3)
        self.assertEqual(client.datasets[0]["name"], DATASET)
        self.assertEqual(client.datasets[0]["metadata"], {"golden_sha256_16": "abc"})
        self.assertEqual([i["id"] for i in client.items], [item_id(r["id"]) for r in ROWS])
        self.assertTrue(all(i["dataset_name"] == DATASET for i in client.items))
        self.assertTrue(client.flushed)

    def test_sync_without_keys_fails_before_calling_langfuse(self):
        env = {k: "" for k in tracing.KEY_VARS}
        with (mock.patch.dict(os.environ, env), mock.patch.object(tracing, "load_dotenv"),
              mock.patch.object(tracing, "get_client") as get_client, self.assertRaises(RuntimeError)):
            langfuse_sync.main()
        get_client.assert_not_called()


class TracedRunTest(_Base):
    cfg = RunConfig(name="test", mode="hybrid_rrf", context_k=2)

    def test_traced_run_matches_untraced_and_posts_scores(self):
        dataset = FakeDataset(list(reversed(_items(ROWS))) + [SimpleNamespace(id="other-item")])
        result, experiment = run_traced(ROWS, self.cfg, self._services(), load_prices(), dataset, "test 2026")
        self.assertEqual([r["id"] for r in result["rows"]], ["g001", "g002", "g003"], "golden order, not dataset order")
        self.assertEqual(dataset.calls, [{"name": "test", "run_name": "test 2026", "max_concurrency": 1}])
        self.assertEqual(len(dataset.items), 3, "items outside the selected rows are not run")
        scores = {e.name: e.value for e in dataset.scores[item_id("g001")]}
        self.assertEqual(scores["hit@5"], 1.0)
        self.assertEqual(scores["refused"], 0.0)
        self.assertNotIn("hit@5", {e.name for e in dataset.scores[item_id("g003")]})
        self.assertEqual(experiment.experiment_url, "https://example.invalid/run")

        # the traced rows are the same rows an untraced run produces
        with tempfile.TemporaryDirectory() as tmp:
            self.cache.close()
            self.cache = DiskCache(Path(tmp) / "c.sqlite")
            plain = run(ROWS, self.cfg, self._services(), load_prices())
        self.assertEqual(_without_latency(result["rows"]), _without_latency(plain["rows"]))

    def test_rows_missing_from_the_dataset_stop_the_run(self):
        dataset = FakeDataset(_items(ROWS[:1]))
        with self.assertRaises(SystemExit):
            run_traced(ROWS, self.cfg, self._services(), load_prices(), dataset, "x")
        self.assertEqual(dataset.calls, [])

    def test_trace_flag_without_keys_fails_before_any_paid_call(self):
        env = {k: "" for k in tracing.KEY_VARS}
        with (mock.patch.dict(os.environ, env), mock.patch.object(tracing, "load_dotenv"),
              mock.patch("eval.run.real_services") as services, self.assertRaises(RuntimeError)):
            main(["--config", "dense", "--limit", "1", "--trace"])
        services.assert_not_called()


if __name__ == "__main__":
    unittest.main()
