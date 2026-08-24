"""End-to-end pipeline CLI.

Usage (from the repository root, after configuring AWS credentials):

    python -m src.rag_pipeline.pipeline ingest
    python -m src.rag_pipeline.pipeline ask "How many weeks of parental leave do we get?"
    python -m src.rag_pipeline.pipeline ask "..." --tenant acme --top-k 5
    python -m src.rag_pipeline.pipeline rerank-demo "What if I leave after education reimbursement?"
    python -m src.rag_pipeline.pipeline evaluate
    python -m src.rag_pipeline.pipeline cleanup --confirm

Set VECTOR_BUCKET_NAME in the environment to reuse the same bucket across
runs; otherwise a unique demo bucket name is generated per process.
"""

import argparse
import sys

sys.path.insert(0, ".")  # allow `python -m src.rag_pipeline.pipeline` from repo root

from data.sample_corpus import DOCUMENTS  # noqa: E402

from . import config  # noqa: E402
from .chunking import chunk_documents  # noqa: E402
from .cleanup import cleanup_all  # noqa: E402
from .embeddings import embed_chunks  # noqa: E402
from .generation import answer_from_retrieved_chunks, generate_answer  # noqa: E402
from .rerank import rerank_with_llm  # noqa: E402
from .vector_store import ensure_index, ensure_vector_bucket, ingest_chunks, query_index  # noqa: E402


def cmd_ingest(_args) -> None:
    print(f"Bucket: {config.VECTOR_BUCKET_NAME} | Index: {config.VECTOR_INDEX_NAME}")
    print(f"Chunking: {config.CHUNKING_VERSION}\n")
    ensure_vector_bucket()
    ensure_index()
    chunks = chunk_documents(
        DOCUMENTS,
        strategy=config.CHUNK_STRATEGY,
        chunk_size=config.CHUNK_SIZE,
        overlap=config.CHUNK_OVERLAP,
        chunking_version=config.CHUNKING_VERSION,
    )
    print(f"Chunked {len(DOCUMENTS)} documents into {len(chunks)} chunks. Embedding...")
    embed_chunks(chunks)
    ingest_chunks(chunks)
    print("\nIngestion complete.")


def cmd_ask(args) -> None:
    metadata_filter = {"tenant_id": {"$eq": args.tenant}} if args.tenant else None
    result = generate_answer(args.question, top_k=args.top_k, metadata_filter=metadata_filter)
    print(f"\nQ: {result['question']}\n")
    print(f"A: {result['answer']}\n")
    print("Retrieved context:")
    for i, chunk in enumerate(result["retrieved_chunks"], 1):
        meta = chunk.get("metadata", {})
        print(
            f"  [{i}] distance={chunk.get('distance'):.4f}  "
            f"{meta.get('title')} (p.{meta.get('page')})"
        )


def cmd_rerank_demo(args) -> None:
    candidates = query_index(args.question, top_k=10)
    print(f"Retrieved {len(candidates)} candidates by vector similarity.\n")
    reranked = rerank_with_llm(args.question, candidates, keep_top=3)
    print("=== Answer from vector-order top 3 ===")
    print(answer_from_retrieved_chunks(args.question, candidates[:3])["answer"])
    print("\n=== Answer from reranked top 3 ===")
    print(answer_from_retrieved_chunks(args.question, reranked)["answer"])


def cmd_evaluate(_args) -> None:
    from .evaluate import no_answer_distance_probe, retrieval_accuracy

    report = retrieval_accuracy(top_k=3)
    print(f"Top-{report['top_k']} top-doc retrieval accuracy: {report['accuracy']:.0%}\n")
    for d in report["details"]:
        flag = "PASS" if d["hit"] else "MISS"
        print(f"  [{flag}] {d['question']}")
    print()
    no_answer_distance_probe()


def cmd_cost(_args) -> None:
    from .cost_model import estimate_rag_cost, print_worksheet

    print_worksheet(estimate_rag_cost())


def cmd_cleanup(args) -> None:
    cleanup_all(confirm=args.confirm)


def main() -> None:
    parser = argparse.ArgumentParser(description="RAG on AWS with S3 Vectors and Bedrock")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("ingest", help="Chunk, embed and index the sample corpus").set_defaults(func=cmd_ingest)

    ask = sub.add_parser("ask", help="Ask a grounded question")
    ask.add_argument("question")
    ask.add_argument("--top-k", type=int, default=config.DEFAULT_TOP_K)
    ask.add_argument("--tenant", default=None, help="Enforce tenant isolation via metadata filter")
    ask.set_defaults(func=cmd_ask)

    rr = sub.add_parser("rerank-demo", help="Compare vector order vs LLM-reranked answers")
    rr.add_argument("question")
    rr.set_defaults(func=cmd_rerank_demo)

    sub.add_parser("evaluate", help="Run the retrieval evaluation set").set_defaults(func=cmd_evaluate)

    sub.add_parser("cost", help="Print the unit-economics worksheet").set_defaults(func=cmd_cost)

    cl = sub.add_parser("cleanup", help="Delete the demo index and vector bucket")
    cl.add_argument("--confirm", action="store_true")
    cl.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
