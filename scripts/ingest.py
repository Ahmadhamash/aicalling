"""Embed and upsert business knowledge-base data into Qdrant.

Usage
-----
    python -m scripts.ingest                 # ingest the bundled sample data
    python -m scripts.ingest path/to/kb.json # ingest your own data file

The input JSON is a list of records::

    [
      {
        "id": "hours-1",                 # optional; auto-generated if absent
        "text": "دوامنا من التاسعة ...",  # required — the fact to retrieve
        "category": "hours",             # optional metadata (used for prefilter)
        "lang": "ar"                     # optional metadata
      },
      ...
    ]

Every field other than ``text`` is stored verbatim in the point payload and is
available as a metadata prefilter at query time.
"""

from __future__ import annotations

import asyncio
import json
import sys
import uuid
from pathlib import Path
from typing import Any

from openai import AsyncOpenAI
from qdrant_client import models

from app.config import settings
from app.services import vector_db

_DEFAULT_DATA = Path(__file__).parent / "business_data.json"
# Embeddings API allows batching; keep batches modest to stay under token caps.
_BATCH = 64


def _load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as fh:
        data = json.load(fh)
    if not isinstance(data, list):
        raise ValueError("input JSON must be a list of records")
    records = [r for r in data if r.get("text")]
    if not records:
        raise ValueError("no records with a non-empty 'text' field")
    return records


async def _embed_batch(client: AsyncOpenAI, texts: list[str]) -> list[list[float]]:
    resp = await client.embeddings.create(model=settings.embedding_model, input=texts)
    return [item.embedding for item in resp.data]


def _point_id(record: dict[str, Any]) -> str:
    """Stable UUID for a record (derived from its id, or random)."""

    raw = str(record.get("id") or uuid.uuid4())
    return str(uuid.uuid5(uuid.NAMESPACE_URL, raw))


async def ingest(path: Path) -> None:
    """Embed all records in ``path`` and upsert them into Qdrant."""

    records = _load_records(path)
    print(f"Loaded {len(records)} records from {path}")

    await vector_db.ensure_collection()
    client = AsyncOpenAI(api_key=settings.openai_api_key, base_url=settings.openai_base_url)
    qdrant = vector_db.get_client()

    total = 0
    for start in range(0, len(records), _BATCH):
        batch = records[start : start + _BATCH]
        vectors = await _embed_batch(client, [r["text"] for r in batch])
        points = [
            models.PointStruct(
                id=_point_id(record),
                vector=vector,
                # Store the full record (incl. text) as the payload.
                payload=record,
            )
            for record, vector in zip(batch, vectors)
        ]
        await qdrant.upsert(collection_name=settings.qdrant_collection, points=points)
        total += len(points)
        print(f"  upserted {total}/{len(records)}")

    await client.close()
    await vector_db.close_client()
    print(f"Done. {total} points in collection '{settings.qdrant_collection}'.")


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else _DEFAULT_DATA
    if not path.exists():
        raise SystemExit(f"data file not found: {path}")
    asyncio.run(ingest(path))


if __name__ == "__main__":
    main()
