from __future__ import annotations

import uuid

import numpy as np
from qdrant_client import QdrantClient, models

from babyinfoquick.age import AgeBand
from babyinfoquick.config import settings
from babyinfoquick.types import Chunk, Scored


def _point_id(chunk_id: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_URL, chunk_id))


class QdrantStore:
    def __init__(self, collection: str, dim: int, url: str | None = None):
        self.collection = collection
        self.dim = dim
        self.client = QdrantClient(
            url=url or settings.qdrant_url, api_key=settings.qdrant_api_key
        )

    def recreate(self) -> None:
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            collection_name=self.collection,
            vectors_config=models.VectorParams(
                size=self.dim, distance=models.Distance.COSINE
            ),
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="age_bands",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )
        self.client.create_payload_index(
            collection_name=self.collection,
            field_name="doc_id",
            field_schema=models.PayloadSchemaType.KEYWORD,
        )

    def upsert(self, chunks: list[Chunk], vectors: np.ndarray, batch_size: int = 128) -> None:
        for start in range(0, len(chunks), batch_size):
            batch = chunks[start : start + batch_size]
            points = [
                models.PointStruct(
                    id=_point_id(chunk.chunk_id),
                    vector=vectors[start + offset].tolist(),
                    payload=chunk.model_dump(mode="json"),
                )
                for offset, chunk in enumerate(batch)
            ]
            self.client.upsert(collection_name=self.collection, points=points, wait=True)

    def count(self) -> int:
        return self.client.count(self.collection, exact=True).count

    def search(
        self, vector: np.ndarray, top_k: int, age_bands: list[AgeBand] | None = None
    ) -> list[Scored]:
        query_filter = None
        if age_bands:
            query_filter = models.Filter(
                must=[
                    models.FieldCondition(
                        key="age_bands",
                        match=models.MatchAny(any=[b.value for b in age_bands]),
                    )
                ]
            )
        response = self.client.query_points(
            collection_name=self.collection,
            query=vector.tolist(),
            limit=top_k,
            query_filter=query_filter,
            with_payload=True,
        )
        return [
            Scored(
                chunk=Chunk.model_validate(point.payload),
                score=point.score,
                dense_score=point.score,
            )
            for point in response.points
        ]
