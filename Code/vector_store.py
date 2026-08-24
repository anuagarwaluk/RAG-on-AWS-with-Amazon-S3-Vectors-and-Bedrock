"""Amazon S3 Vectors: bucket and index lifecycle, ingestion, and query.

Why S3 Vectors for this pipeline:
* Cost: up to ~90% cheaper than dedicated vector engines for large corpora
* Scale: up to 2 billion vectors per index, 10,000 indexes per bucket
* Latency: sub-second cold, ~100 ms warm. Right for enterprise RAG,
  wrong for high-QPS realtime search (use OpenSearch Serverless there)

Design decisions encoded below:
* The raw chunk ``text`` is stored as NON-FILTERABLE metadata. The LLM only
  ever sees text, never vectors, so the text must live next to the vector
  for single-call retrieval. Non-filterable metadata has a higher size limit.
* ``tenant_id`` and ``access_group`` are filterable metadata: retrieval-time
  security is enforced by injecting a server-side metadata filter into every
  query. Never accept that filter from the client.
* Index configuration (dimension, distance metric, non-filterable keys) is
  IMMUTABLE after creation. Changing the embedding model or dimension means
  a new index and a full re-ingest.
"""

import time
from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from . import config
from .aws_clients import (
    as_float32_list,
    client_error_code,
    client_error_summary,
    s3vectors_client,
)
from .embeddings import embed_text

_NOT_FOUND_CODES = {
    "NotFound",
    "NotFoundException",
    "ResourceNotFoundException",
    "NoSuchVectorBucket",
    "NoSuchIndex",
    "NoSuchVectorIndex",
}


# ---------------------------------------------------------------------------
# Bucket and index lifecycle
# ---------------------------------------------------------------------------
def ensure_vector_bucket(bucket_name: str = config.VECTOR_BUCKET_NAME) -> None:
    """Create the vector bucket if it does not exist, then wait until ready."""
    s3v = s3vectors_client()
    args: Dict[str, Any] = {
        "vectorBucketName": bucket_name,
        "encryptionConfiguration": {"sseType": "AES256"},
        # For a customer-managed key (regulated workloads):
        # "encryptionConfiguration": {"sseType": "aws:kms", "kmsKeyArn": "arn:aws:kms:..."},
    }
    try:
        s3v.create_vector_bucket(**args)
        print(f"Created vector bucket: {bucket_name}")
    except s3v.exceptions.ConflictException:
        print(f"Vector bucket {bucket_name} already exists. Continuing.")
    except ClientError as exc:
        raise RuntimeError(f"Error creating vector bucket: {client_error_summary(exc)}") from exc
    _wait(lambda: s3v.get_vector_bucket(vectorBucketName=bucket_name), what=f"bucket {bucket_name}")


def ensure_index(
    bucket_name: str = config.VECTOR_BUCKET_NAME,
    index_name: str = config.VECTOR_INDEX_NAME,
    dimensions: int = config.EMBEDDING_DIMENSIONS,
) -> None:
    """Create the vector index if it does not exist, then wait until ready."""
    s3v = s3vectors_client()
    args: Dict[str, Any] = {
        "vectorBucketName": bucket_name,
        "indexName": index_name,
        "dataType": config.VECTOR_DATA_TYPE,
        "dimension": dimensions,
        "distanceMetric": config.DISTANCE_METRIC,
        "metadataConfiguration": {
            "nonFilterableMetadataKeys": config.NON_FILTERABLE_METADATA_KEYS
        },
    }
    try:
        s3v.create_index(**args)
        print(f"Created vector index: {index_name} (dim={dimensions}, metric={config.DISTANCE_METRIC})")
    except s3v.exceptions.ConflictException:
        print(f"Vector index {index_name} already exists. Continuing.")
    except ClientError as exc:
        raise RuntimeError(f"Error creating vector index: {client_error_summary(exc)}") from exc
    _wait(
        lambda: s3v.get_index(vectorBucketName=bucket_name, indexName=index_name),
        what=f"index {index_name}",
    )


def _wait(probe, what: str, timeout_seconds: int = 180, interval: int = 3) -> None:
    deadline = time.time() + timeout_seconds
    while True:
        try:
            probe()
            return
        except ClientError as exc:
            if client_error_code(exc) in _NOT_FOUND_CODES and time.time() < deadline:
                time.sleep(interval)
                continue
            raise RuntimeError(f"Timed out or failed waiting for {what}") from exc


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------
def to_vector_record(chunk: Dict[str, Any]) -> Dict[str, Any]:
    """Shape one chunk into the S3 Vectors record format: key, data, metadata."""
    return {
        "key": chunk["chunk_id"],
        "data": {
            "float32": as_float32_list(
                chunk["embedding"], expected_dimensions=config.EMBEDDING_DIMENSIONS
            )
        },
        "metadata": {
            "text": chunk["text"],                  # non-filterable: what the LLM sees
            "doc_id": chunk["doc_id"],              # filterable
            "title": chunk["title"],                # filterable: citation display
            "source": chunk["source"],              # filterable: scoped retrieval
            "page": chunk["page"],                  # filterable: citation display
            "tenant_id": chunk["tenant_id"],        # filterable: multi-tenant isolation
            "access_group": chunk["access_group"],  # filterable: row-level access
            "chunk_index": chunk["chunk_index"],
            "chunking_version": chunk["chunking_version"],
        },
    }


def ingest_chunks(
    chunks: List[Dict[str, Any]],
    bucket_name: str = config.VECTOR_BUCKET_NAME,
    index_name: str = config.VECTOR_INDEX_NAME,
) -> int:
    """Write embedded chunks into the index in batches (limit: 500/call)."""
    s3v = s3vectors_client()
    records = [to_vector_record(c) for c in chunks]
    written = 0
    for i in range(0, len(records), config.PUT_VECTORS_BATCH_SIZE):
        batch = records[i : i + config.PUT_VECTORS_BATCH_SIZE]
        try:
            s3v.put_vectors(
                vectorBucketName=bucket_name,
                indexName=index_name,
                vectors=batch,
            )
        except ClientError as exc:
            raise RuntimeError(
                f"put_vectors failed on batch starting at {i}: {client_error_summary(exc)}"
            ) from exc
        written += len(batch)
        print(f"Ingested {written}/{len(records)} vectors")
    return written


# ---------------------------------------------------------------------------
# Query
# ---------------------------------------------------------------------------
def query_index(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    metadata_filter: Optional[Dict[str, Any]] = None,
    bucket_name: str = config.VECTOR_BUCKET_NAME,
    index_name: str = config.VECTOR_INDEX_NAME,
) -> List[Dict[str, Any]]:
    """Semantic search: embed the question and return the top-k nearest chunks.

    ``metadata_filter`` example for multi-tenant isolation (inject server side,
    never accept from the client)::

        {"tenant_id": {"$eq": "acme"}}

    Returns a list of dicts with ``key``, ``distance`` (lower = more similar;
    for cosine, distance = 1 - similarity), and ``metadata``.
    """
    if not question or not question.strip():
        raise ValueError("Question cannot be empty.")
    top_k = max(1, min(int(top_k), 100))

    # CRITICAL: identical embedding settings to index time.
    query_vector = embed_text(question, dimensions=config.EMBEDDING_DIMENSIONS, normalize=True)

    kwargs: Dict[str, Any] = {
        "vectorBucketName": bucket_name,
        "indexName": index_name,
        "topK": top_k,
        "queryVector": {
            "float32": as_float32_list(query_vector, expected_dimensions=config.EMBEDDING_DIMENSIONS)
        },
        "returnDistance": True,
        "returnMetadata": True,
    }
    if metadata_filter is not None:
        kwargs["filter"] = metadata_filter

    try:
        response = s3vectors_client().query_vectors(**kwargs)
    except ClientError as exc:
        if client_error_code(exc) in {"AccessDenied", "AccessDeniedException", "Forbidden"}:
            raise RuntimeError(
                "QueryVectors denied. With returnMetadata=True and metadata filters, the "
                "runtime role needs BOTH s3vectors:QueryVectors and s3vectors:GetVectors."
            ) from exc
        raise
    return response.get("vectors", [])
