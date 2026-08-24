"""Grounded answer generation via the Bedrock Converse API.

Two production behaviours are encoded here that separate a demo from a
deployable system:

1. **Grounded prompting with citations.** The LLM is instructed to answer
   ONLY from the retrieved context, to cite sources by number, and to say
   "I don't know" when the context does not contain the answer.

2. **Distance-threshold refusal.** When even the nearest retrieved chunk is
   far from the question (cosine distance above a tuned threshold), the
   pipeline refuses BEFORE calling the LLM. Retrieval distance is a
   hallucination tripwire: a weak top hit means the corpus does not answer
   this question, and no prompt engineering fixes retrieving the wrong text.

The Converse API is used instead of provider-specific invoke bodies so the
generation model can be swapped (Claude, Nova, others) without code changes.
"""

from typing import Any, Dict, List, Optional

from botocore.exceptions import ClientError

from . import config
from .aws_clients import bedrock_runtime_client
from .vector_store import query_index

_NO_ANSWER = "I don't have that information in the current knowledge base."

_resolved_model_id: Optional[str] = None


def _generation_model_id() -> str:
    """Resolve a working generation model once, preferring the configured one."""
    global _resolved_model_id
    if _resolved_model_id:
        return _resolved_model_id
    candidates = (
        [config.GENERATION_MODEL_ID] if config.GENERATION_MODEL_ID else []
    ) + config.PREFERRED_GENERATION_MODEL_IDS
    failures = []
    for candidate in candidates:
        try:
            invoke_generation_model("ping", max_tokens=5, model_id=candidate)
            _resolved_model_id = candidate
            return candidate
        except ClientError as exc:
            failures.append(f"{candidate}: {exc.response.get('Error', {}).get('Code')}")
    raise RuntimeError(
        "No generation model is accessible. Enable model access in the Bedrock "
        f"console or set BEDROCK_GENERATION_MODEL_ID. Tried: {failures}"
    )


def invoke_generation_model(
    prompt: str,
    max_tokens: int = config.MAX_ANSWER_TOKENS,
    temperature: float = config.GENERATION_TEMPERATURE,
    model_id: Optional[str] = None,
) -> str:
    """Generate text with the provider-agnostic Converse API."""
    response = bedrock_runtime_client().converse(
        modelId=model_id or _generation_model_id(),
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": max_tokens, "temperature": temperature},
    )
    return response["output"]["message"]["content"][0]["text"]


def build_rag_prompt(question: str, retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Assemble the grounded prompt: numbered, cited context + guardrail rules."""
    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, 1):
        meta = chunk.get("metadata", {})
        context_blocks.append(
            f"[Source {i}] (document: {meta.get('title', 'unknown')}, "
            f"source: {meta.get('source', 'unknown')}, page: {meta.get('page', '?')})\n"
            f"{meta.get('text', '')}\n"
        )
    context = "\n".join(context_blocks)
    return f"""You are an assistant for company policy questions. Answer the question using ONLY the context below.

If the answer is not contained in the context, say "{_NO_ANSWER}" Do not make up details.
When you cite information, reference the source by number in square brackets, like [Source 1].

CONTEXT:
{context}

QUESTION: {question}

ANSWER:"""


def answer_from_retrieved_chunks(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    max_tokens: int = config.MAX_ANSWER_TOKENS,
) -> Dict[str, Any]:
    """Generate from an already retrieved (or reranked) context list."""
    if not retrieved_chunks:
        return {"question": question, "answer": _NO_ANSWER, "retrieved_chunks": [], "prompt_length_chars": 0}
    prompt = build_rag_prompt(question, retrieved_chunks)
    answer_text = invoke_generation_model(prompt, max_tokens=max_tokens)
    return {
        "question": question,
        "answer": answer_text,
        "retrieved_chunks": retrieved_chunks,
        "prompt_length_chars": len(prompt),
    }


def generate_answer(
    question: str,
    top_k: int = config.DEFAULT_TOP_K,
    metadata_filter: Optional[Dict[str, Any]] = None,
    distance_threshold: Optional[float] = None,
) -> Dict[str, Any]:
    """End-to-end RAG: Retrieve, apply the no-answer tripwire, Augment, Generate."""
    # Retrieval
    retrieved = query_index(question, top_k=top_k, metadata_filter=metadata_filter)
    if not retrieved:
        return {"question": question, "answer": _NO_ANSWER, "retrieved_chunks": [], "prompt_length_chars": 0}

    # Hallucination tripwire: weakest acceptable top-hit distance
    threshold = config.NO_ANSWER_DISTANCE_THRESHOLD if distance_threshold is None else distance_threshold
    top_distance = retrieved[0].get("distance")
    if threshold is not None and isinstance(top_distance, (int, float)) and top_distance > threshold:
        return {
            "question": question,
            "answer": _NO_ANSWER,
            "retrieved_chunks": retrieved,
            "prompt_length_chars": 0,
        }

    # Augmentation + generation
    return answer_from_retrieved_chunks(question, retrieved)
