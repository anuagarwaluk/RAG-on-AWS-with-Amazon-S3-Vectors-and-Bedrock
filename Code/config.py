"""Central configuration for the RAG pipeline.

Every tunable lives here so the high-leverage knobs (chunking, embedding
model, top-k, distance threshold) can be changed in one place and every
module stays consistent. The rule that matters most: the embedding model,
dimensions, and normalisation used at INDEX time must be identical at
QUERY time, or similarity search quietly returns junk.
"""

import os
import uuid

# ---------------------------------------------------------------------------
# AWS
# ---------------------------------------------------------------------------
AWS_REGION = os.environ.get("AWS_REGION", "us-east-1")

# ---------------------------------------------------------------------------
# S3 Vectors resources
# ---------------------------------------------------------------------------
_UNIQUE_SUFFIX = uuid.uuid4().hex[:8]
VECTOR_BUCKET_NAME = os.environ.get("VECTOR_BUCKET_NAME") or f"rag-demo-{_UNIQUE_SUFFIX}"
VECTOR_INDEX_NAME = os.environ.get("VECTOR_INDEX_NAME", "hr-policy-index")

# Immutable at index creation time. Changing any of these later means
# creating a NEW index and re-ingesting. Plan for this footgun.
DISTANCE_METRIC = "cosine"          # cosine or euclidean
VECTOR_DATA_TYPE = "float32"        # only float32 today
NON_FILTERABLE_METADATA_KEYS = ["text"]  # raw chunk text: retrievable, never filtered on

# ---------------------------------------------------------------------------
# Models (Amazon Bedrock)
# ---------------------------------------------------------------------------
EMBEDDING_MODEL_ID = os.environ.get(
    "BEDROCK_EMBEDDING_MODEL_ID", "amazon.titan-embed-text-v2:0"
)
EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "1024"))  # 256 | 512 | 1024

# Generation model resolved at runtime via the Converse API so the pipeline
# is not tied to one provider's request body. Override to pin a model.
GENERATION_MODEL_ID = os.environ.get("BEDROCK_GENERATION_MODEL_ID") or None
PREFERRED_GENERATION_MODEL_IDS = [
    "us.anthropic.claude-haiku-4-5-20251001-v1:0",
    "us.anthropic.claude-sonnet-4-6",
    "amazon.nova-lite-v1:0",
    "amazon.nova-pro-v1:0",
]

# ---------------------------------------------------------------------------
# Chunking knobs (highest-leverage retrieval quality lever)
# ---------------------------------------------------------------------------
CHUNK_STRATEGY = os.environ.get("CHUNK_STRATEGY", "recursive")  # recursive | fixed
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "400"))           # characters; try 200 / 400 / 800
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "50"))      # characters; try 0 / 50 / 100
CHUNKING_VERSION = f"{CHUNK_STRATEGY}-chars-size{CHUNK_SIZE}-overlap{CHUNK_OVERLAP}"

# ---------------------------------------------------------------------------
# Retrieval and generation knobs
# ---------------------------------------------------------------------------
DEFAULT_TOP_K = 5                   # S3 Vectors allows up to 100
PUT_VECTORS_BATCH_SIZE = 500        # hard service limit per PutVectors call

# If the nearest match is further than this cosine distance, answer
# "I don't know" instead of letting the LLM improvise. Tune per corpus
# by inspecting distances for questions the corpus cannot answer.
NO_ANSWER_DISTANCE_THRESHOLD = 0.62

MAX_ANSWER_TOKENS = 500
GENERATION_TEMPERATURE = 0.0        # deterministic answers for grounded Q&A
