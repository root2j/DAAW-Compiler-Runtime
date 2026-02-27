"""Artifact store — async-safe, JSON-persisted key-value store."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any


class ArtifactStore:
    """Coroutine-safe key-value store with JSON persistence."""

    def __init__(self, persist_dir: str = ".daaw_store"):
        self._persist_dir = persist_dir
        self._file = os.path.join(persist_dir, "artifacts.json")
        self._data: dict[str, Any] = {}
        self._lock = asyncio.Lock()
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self._file):
            with open(self._file, "r", encoding="utf-8") as f:
                self._data = json.load(f)

    async def _persist(self) -> None:
        os.makedirs(self._persist_dir, exist_ok=True)

        def _write() -> None:
            with open(self._file, "w", encoding="utf-8") as f:
                json.dump(self._data, f, indent=2, default=str)

        await asyncio.to_thread(_write)

    async def put(self, key: str, value: Any) -> None:
        async with self._lock:
            self._data[key] = value
            await self._persist()

    async def get(self, key: str, default: Any = None) -> Any:
        async with self._lock:
            return self._data.get(key, default)

    async def get_many(self, keys: list[str]) -> dict[str, Any]:
        async with self._lock:
            return {k: self._data[k] for k in keys if k in self._data}

    async def get_namespace(self, prefix: str) -> dict[str, Any]:
        async with self._lock:
            return {k: v for k, v in self._data.items() if k.startswith(prefix)}

    async def delete(self, key: str) -> None:
        async with self._lock:
            self._data.pop(key, None)
            await self._persist()

    async def clear(self) -> None:
        async with self._lock:
            self._data.clear()
            await self._persist()
