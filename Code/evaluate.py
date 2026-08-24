"""Retrieval evaluation and knob experiments.

RAG quality work without an evaluation set is guesswork. This module keeps a
tiny labelled set of (question, expected_doc_id) pairs and measures top-doc
retrieval accuracy: does the correct source document appear in the top-k?

It also demonstrates the two most instructive sensitivity experiments:

* **top-k sweep**: too low misses the answer, too high dilutes the prompt
  with weakly relevant text and can degrade the final answer.
* **distance inspection**: for questions the corpus cannot answer, the
  nearest-hit distance jumps. That gap is how the no-answer threshold in
  ``generation.py`` gets tuned.
"""

from typing import Any, Dict, List, Tuple

from . import config
from .generation import generate_answer
from .vector_store import query_index

# (question, doc_id expected as the top retrieved document)
EVAL_SET: List[Tuple[str, str]] = [
    ("How many weeks of parental leave do primary caregivers get?", "hr-parental-leave"),
    ("How many days of annual leave can I carry over?", "hr-annual-leave"),
    ("Can I work from another country for a month?", "hr-remote-work"),
    ("What happens to education reimbursement if I resign?", "hr-education"),
    ("What is the client entertainment limit per head?", "hr-expenses"),
]


def retrieval_accuracy(top_k: int = 3) -> Dict[str, Any]:
    """Fraction of eval questions whose expected document is retrieved in top-k."""
    hits, details = 0, []
    for question, expected_doc in EVAL_SET:
        retrieved = query_index(question, top_k=top_k)
        retrieved_docs = [r.get("metadata", {}).get("doc_id") for r in retrieved]
        hit = expected_doc in retrieved_docs
        hits += int(hit)
        details.append(
            {
                "question": question,
                "expected": expected_doc,
                "retrieved": retrieved_docs,
                "top_distance": retrieved[0].get("distance") if retrieved else None,
                "hit": hit,
            }
        )
    return {"top_k": top_k, "accuracy": hits / len(EVAL_SET), "details": details}


def top_k_sweep(question: str, ks: List[int] = [1, 3, 5, 10]) -> None:
    """Show how the final answer changes as top_k grows."""
    for k in ks:
        result = generate_answer(question, top_k=k)
        print(f"\n=== top_k={k} | prompt={result['prompt_length_chars']} chars ===")
        print(result["answer"])


def no_answer_distance_probe(
    unanswerable_question: str = "What is our policy on free lunches?",
) -> None:
    """Inspect distances for a question the corpus cannot answer.

    Compare the top-hit distance here against answerable questions; set
    NO_ANSWER_DISTANCE_THRESHOLD between the two bands.
    """
    results = query_index(unanswerable_question, top_k=3)
    print(f"Question with no answer in corpus: {unanswerable_question}")
    for i, r in enumerate(results, 1):
        meta = r.get("metadata", {})
        print(f"  {i}. distance={r.get('distance'):.4f}  doc={meta.get('doc_id')}")
    print(f"Current threshold: {config.NO_ANSWER_DISTANCE_THRESHOLD}")


if __name__ == "__main__":
    report = retrieval_accuracy(top_k=3)
    print(f"Top-{report['top_k']} top-doc retrieval accuracy: {report['accuracy']:.0%}")
    for d in report["details"]:
        flag = "PASS" if d["hit"] else "MISS"
        print(f"  [{flag}] {d['question']}  (top distance {d['top_distance']:.4f})")
    print()
    no_answer_distance_probe()
