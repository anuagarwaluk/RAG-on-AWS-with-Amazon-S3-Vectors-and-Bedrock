"""Embedding generation via Amazon Bedrock (Titan Text Embeddings V2).

The single most important rule in RAG: the SAME embedding model, dimension,
and normalisation setting must be used at index time and at query time.
Break that and similarity search silently degrades into noise; nothing
errors, the answers just get worse.

Titan Text Embeddings V2 characteristics:
* 1024 dimensions by default, configurable to 256 or 512 for storage savings
* Normalised output, which pairs well with cosine distance
* Multilingual support

If the corpus is in a specialised domain (legal, medical, finance), evaluate
domain-tuned embedding models: on the same documents and chunks, a domain
embedding model can move retrieval accuracy dramatically. The embedding model
is the second-highest leverage knob after chunking.
"""

import json
from typing import Any, Dict, List

from . import config
from .aws_clients import as_float32_list, bedrock_runtime_client


def embed_text(
    text: str,
    dimensions: int = config.EMBEDDING_DIMENSIONS,
    normalize: bool = True,
) -> List[float]:
    """Embed a single string with Titan Text Embeddings V2."""
    if not text or not text.strip():
        raise ValueError("Cannot embed empty text.")
    response = bedrock_runtime_client().invoke_model(
        modelId=config.EMBEDDING_MODEL_ID,
        body=json.dumps(
            {
                "inputText": text,
                "dimensions": dimensions,
                "normalize": normalize,
            }
        ),
        accept="application/json",
        contentType="application/json",
    )
    body = json.loads(response["body"].read())
    embedding = body.get("embedding")
    if embedding is None:
        # Defensive fallback for alternate embedding response shapes.
        embedding = (body.get("embeddingsByType") or {}).get("float")
    if embedding is None:
        raise RuntimeError(
            f"Embedding response did not include an embedding. Keys: {list(body.keys())}"
        )
    return as_float32_list(embedding, expected_dimensions=dimensions)


def embed_chunks(chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Attach an ``embedding`` field to each chunk record in place."""
    for chunk in chunks:
        chunk["embedding"] = embed_text(chunk["text"])
    return chunks
