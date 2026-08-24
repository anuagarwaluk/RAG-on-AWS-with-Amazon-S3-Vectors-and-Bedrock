"""Chunking strategies.

Chunking is the highest-leverage RAG quality knob: split documents wrong and
the right answer never gets retrieved, no matter which embedding model or
LLM sits downstream.

Two strategies are implemented:

* ``fixed_size_chunker``   - fixed character windows with overlap. Fast,
  predictable, ignores structure. Fine for uniform prose.
* ``recursive_chunker``    - splits on paragraphs, then sentences, then
  words, packing units up to the chunk size while preserving overlap.
  Keeps semantic units intact, which is why it is the default.

Both are intentionally character-based for clarity. Production systems
should use a tokenizer-aware chunker and validate settings against an
evaluation set (see ``evaluate.py``).
"""

import hashlib
from typing import Any, Dict, List


def _validate(chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    if overlap < 0 or overlap >= chunk_size:
        raise ValueError("overlap must be >= 0 and smaller than chunk_size")


def fixed_size_chunker(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Slide a fixed character window across the text."""
    _validate(chunk_size, overlap)
    if not text:
        return []
    step = chunk_size - overlap
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), step)]
    return [c for c in chunks if c.strip()]


def _split_to_units(text: str, max_unit_size: int, separators: List[str]) -> List[str]:
    """Recursively split text into units no larger than max_unit_size."""
    if len(text) <= max_unit_size or not separators:
        return [text]
    sep, rest = separators[0], separators[1:]
    if sep == "":
        return [text[i : i + max_unit_size] for i in range(0, len(text), max_unit_size)]
    parts = text.split(sep)
    units: List[str] = []
    for i, part in enumerate(parts):
        piece = part + (sep if i < len(parts) - 1 else "")
        if len(piece) <= max_unit_size:
            if piece.strip():
                units.append(piece)
        else:
            units.extend(_split_to_units(piece, max_unit_size, rest))
    return units


def _pack_units_with_overlap(units: List[str], chunk_size: int, overlap: int) -> List[str]:
    """Greedily pack units into chunks, carrying a character overlap forward."""
    chunks: List[str] = []
    current = ""
    for unit in units:
        candidate = current + unit
        if len(candidate) <= chunk_size or not current:
            current = candidate
        else:
            chunks.append(current)
            tail = current[-overlap:] if overlap else ""
            current = tail + unit
    if current.strip():
        chunks.append(current)
    return chunks


def recursive_chunker(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Structure-aware chunking: paragraphs, then sentences, then words."""
    _validate(chunk_size, overlap)
    if not text:
        return []
    separators = ["\n\n", ". ", " ", ""]
    units = _split_to_units(text, max_unit_size=chunk_size, separators=separators)
    return _pack_units_with_overlap(units, chunk_size=chunk_size, overlap=overlap)


def deterministic_chunk_id(doc_id: str, text: str, chunking_version: str) -> str:
    """Stable chunk key: re-ingestion is idempotent, and the key changes
    whenever the content or the chunking strategy changes."""
    digest = hashlib.sha256(f"{doc_id}|{chunking_version}|{text}".encode("utf-8")).hexdigest()[:16]
    return f"{doc_id}-{digest}"


def chunk_documents(
    documents: List[Dict[str, Any]],
    strategy: str = "recursive",
    chunk_size: int = 400,
    overlap: int = 50,
    chunking_version: str = "recursive-chars-size400-overlap50",
) -> List[Dict[str, Any]]:
    """Chunk a document corpus into records ready for embedding.

    Every chunk carries the metadata that later enables citations
    (title, source, page) and retrieval security (tenant_id, access_group).
    """
    chunker = recursive_chunker if strategy == "recursive" else fixed_size_chunker
    all_chunks: List[Dict[str, Any]] = []
    for doc in documents:
        pieces = chunker(doc["text"], chunk_size=chunk_size, overlap=overlap)
        for i, piece in enumerate(pieces):
            all_chunks.append(
                {
                    "chunk_id": deterministic_chunk_id(doc["doc_id"], piece, chunking_version),
                    "text": piece,
                    "doc_id": doc["doc_id"],
                    "title": doc["title"],
                    "source": doc["source"],
                    "page": doc["page"],
                    "tenant_id": doc["tenant_id"],
                    "access_group": doc["access_group"],
                    "chunk_index": i,
                    "chunking_version": chunking_version,
                }
            )
    return all_chunks
