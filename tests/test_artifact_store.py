"""Tests for the artifact store."""

import asyncio
import json
import os

import pytest

from daaw.store.artifact_store import ArtifactStore


def run(coro):
    """Helper to run async code in sync tests."""
    return asyncio.get_event_loop().run_until_complete(coro)


class TestArtifactStore:
    def test_put_and_get(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("key1", "value1"))
        assert run(store.get("key1")) == "value1"

    def test_get_default(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        assert run(store.get("missing")) is None
        assert run(store.get("missing", "default")) == "default"

    def test_get_many(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("a", 1))
        run(store.put("b", 2))
        run(store.put("c", 3))

        result = run(store.get_many(["a", "c", "missing"]))
        assert result == {"a": 1, "c": 3}

    def test_get_namespace(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("task_001.output", "result"))
        run(store.put("task_001.status", "success"))
        run(store.put("task_002.output", "other"))

        ns = run(store.get_namespace("task_001."))
        assert ns == {"task_001.output": "result", "task_001.status": "success"}

    def test_delete(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("key", "val"))
        run(store.delete("key"))
        assert run(store.get("key")) is None

    def test_delete_nonexistent_is_noop(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.delete("never_existed"))  # should not raise

    def test_clear(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("a", 1))
        run(store.put("b", 2))
        run(store.clear())
        assert run(store.get("a")) is None
        assert run(store.get("b")) is None

    def test_persistence(self, tmp_store_dir):
        """Data should survive a new ArtifactStore instance pointed at same dir."""
        store1 = ArtifactStore(tmp_store_dir)
        run(store1.put("persistent_key", {"nested": "data"}))

        # Create a new instance — should load from disk
        store2 = ArtifactStore(tmp_store_dir)
        result = run(store2.get("persistent_key"))
        assert result == {"nested": "data"}

    def test_persistence_file_exists(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("x", 42))

        filepath = os.path.join(tmp_store_dir, "artifacts.json")
        assert os.path.isfile(filepath)

        with open(filepath) as f:
            data = json.load(f)
        assert data["x"] == 42

    def test_complex_values(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("list", [1, 2, 3]))
        run(store.put("dict", {"a": {"b": "c"}}))
        run(store.put("none", None))

        assert run(store.get("list")) == [1, 2, 3]
        assert run(store.get("dict")) == {"a": {"b": "c"}}
        assert run(store.get("none")) is None  # stored None vs missing both return None

    def test_overwrite(self, tmp_store_dir):
        store = ArtifactStore(tmp_store_dir)
        run(store.put("key", "old"))
        run(store.put("key", "new"))
        assert run(store.get("key")) == "new"
