"""Two-stage retrieval: wide vector recall, then reranking.

Vector similarity is a blunt instrument: it finds chunks that are ABOUT the
same topic, not necessarily chunks that ANSWER the question. Reranking fixes
this by retrieving a wider candidate pool (for example top 10) and re-scoring
each candidate against the question with a model that reads both.

This module demonstrates the pattern with an LLM-as-reranker, which is easy
to read and needs no extra model access. In production, use a purpose-built
reranker instead: Cohere Rerank on Bedrock, or the built-in reranking in
Bedrock Knowledge Bases. Do not reinvent this wheel at scale.
"""

import json
from typing import Any, Dict, List

from .generation import invoke_generation_model


def rerank_with_llm(
    question: str,
    candidates: List[Dict[str, Any]],
    keep_top: int = 5,
) -> List[Dict[str, Any]]:
    """Re-order candidate chunks by how directly each answers the question."""
    if len(candidates) <= 1:
        return candidates

    numbered = "\n".join(
        f"{i}. {(c.get('metadata', {}).get('text') or '')[:400]}"
        for i, c in enumerate(candidates, 1)
    )
    prompt = f"""You are a search result reranker. Given a question and numbered passages,
return ONLY a JSON array of passage numbers, ordered from most to least relevant
for ANSWERING the question directly. Example: [3, 1, 2]

QUESTION: {question}

PASSAGES:
{numbered}

JSON array:"""

    raw = invoke_generation_model(prompt, max_tokens=100, temperature=0.0)
    try:
        order = json.loads(raw.strip())
        reranked = [candidates[i - 1] for i in order if 1 <= i <= len(candidates)]
        # Preserve anything the model dropped, in original order, then trim.
        seen = {id(c) for c in reranked}
        reranked += [c for c in candidates if id(c) not in seen]
        return reranked[:keep_top]
    except (ValueError, TypeError, IndexError) as exc:
        print(f"Rerank parse failed ({exc}); falling back to vector order.")
        return candidates[:keep_top]
