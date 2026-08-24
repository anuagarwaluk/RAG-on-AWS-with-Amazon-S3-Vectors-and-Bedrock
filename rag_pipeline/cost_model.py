"""Unit economics for the RAG workload.

Architecture reviews that stop at "it works" are half-finished. This module
turns the pipeline into a cost worksheet an executive can interrogate: what
does this system cost per month at N queries per day over an M-chunk corpus,
and which line item dominates?

Structure of the model:
* S3 Vectors: storage (logical GB of vectors + metadata + keys), a one-time
  put charge on ingest, per-query API requests, and query processing. The
  processing line is modelled as a conservative ceiling that assumes every
  query scans the whole index; the real ANN index plus metadata filters scan
  far less, so actual cost lands below this number.
* Bedrock: one-time corpus embedding, per-query question embedding, and
  generation input/output tokens (input dominates because retrieved context
  is stuffed into every prompt; this is why top-k discipline is a cost lever,
  not just a quality lever).

PRICES holds illustrative numbers. Always replace them with current regional
pricing before using this in a real proposal.
"""

from typing import Any, Dict

# Illustrative workload: tune to the engagement.
WORKLOAD: Dict[str, Any] = {
    "num_chunks": 1_000_000,
    "embedding_dimensions": 1024,
    "bytes_per_dimension": 4,           # float32
    "avg_metadata_bytes": 500,
    "avg_key_bytes": 64,
    "queries_per_day": 10_000,
    "days_per_month": 30,
    "avg_corpus_chunk_tokens": 500,
    "avg_query_tokens": 25,
    "avg_generation_input_tokens": 2_500,   # question + retrieved context
    "avg_generation_output_tokens": 300,
}

# Illustrative prices. VERIFY BY REGION before quoting anyone.
PRICES: Dict[str, float] = {
    "s3v_storage_per_gb_month": 0.06,
    "s3v_put_per_logical_gb": 0.20,
    "s3v_query_api_per_million": 2.50,
    "s3v_query_processing_per_tb": 0.004,
    "embedding_per_1m_tokens": 0.02,
    "generation_input_per_1m_tokens": 1.00,
    "generation_output_per_1m_tokens": 5.00,
}


def estimate_rag_cost(
    workload: Dict[str, Any] = WORKLOAD,
    prices: Dict[str, float] = PRICES,
) -> Dict[str, float]:
    """Return one-time and monthly cost lines for the modelled workload."""
    per_vector_bytes = (
        workload["embedding_dimensions"] * workload["bytes_per_dimension"]
        + workload["avg_metadata_bytes"]
        + workload["avg_key_bytes"]
    )
    logical_gb = workload["num_chunks"] * per_vector_bytes / (1024**3)

    monthly_queries = workload["queries_per_day"] * workload["days_per_month"]

    # S3 Vectors
    storage_monthly = logical_gb * prices["s3v_storage_per_gb_month"]
    one_time_put = logical_gb * prices["s3v_put_per_logical_gb"]
    query_api_monthly = (monthly_queries / 1_000_000) * prices["s3v_query_api_per_million"]
    # Conservative ceiling: models each query as scanning the whole index.
    query_processing_monthly = (
        (logical_gb / 1024) * monthly_queries * prices["s3v_query_processing_per_tb"]
    )

    # Bedrock
    one_time_corpus_embedding = (
        workload["num_chunks"] * workload["avg_corpus_chunk_tokens"] / 1_000_000
    ) * prices["embedding_per_1m_tokens"]
    query_embedding_monthly = (
        monthly_queries * workload["avg_query_tokens"] / 1_000_000
    ) * prices["embedding_per_1m_tokens"]
    generation_monthly = (
        (monthly_queries * workload["avg_generation_input_tokens"] / 1_000_000)
        * prices["generation_input_per_1m_tokens"]
        + (monthly_queries * workload["avg_generation_output_tokens"] / 1_000_000)
        * prices["generation_output_per_1m_tokens"]
    )

    monthly_total = (
        storage_monthly
        + query_api_monthly
        + query_processing_monthly
        + query_embedding_monthly
        + generation_monthly
    )
    return {
        "logical_gb": logical_gb,
        "monthly_queries": monthly_queries,
        "s3_vectors_storage_monthly": storage_monthly,
        "s3_vectors_query_api_monthly": query_api_monthly,
        "s3_vectors_query_processing_monthly_ceiling": query_processing_monthly,
        "bedrock_query_embedding_monthly": query_embedding_monthly,
        "bedrock_generation_monthly": generation_monthly,
        "monthly_total_excluding_one_time": monthly_total,
        "one_time_s3_vectors_put": one_time_put,
        "one_time_corpus_embedding": one_time_corpus_embedding,
    }


def print_worksheet(estimate: Dict[str, float]) -> None:
    print("RAG cost worksheet (illustrative prices; verify by region)")
    print("-" * 58)
    for key, value in estimate.items():
        if key in {"logical_gb", "monthly_queries"}:
            print(f"{key:48s}: {value:,.2f}")
        else:
            print(f"{key:48s}: ${value:,.2f}")
    print(
        "\nObservation: generation input tokens usually dominate. Retrieval "
        "discipline (top-k, chunk size) is a cost lever as well as a quality lever."
    )


if __name__ == "__main__":
    print_worksheet(estimate_rag_cost())
