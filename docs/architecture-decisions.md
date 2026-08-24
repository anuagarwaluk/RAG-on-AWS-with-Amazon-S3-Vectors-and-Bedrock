# Architecture Decision Notes

The build in this repository is one point in a design space. These notes record the decisions, the alternatives, and when I would choose differently.

## Vector store selection

| Requirement profile | Recommendation |
|---|---|
| Cost-optimised storage for large corpora, sub-second latency acceptable | **Amazon S3 Vectors** (this repo) |
| Sub-100 ms vector queries at high QPS | Amazon OpenSearch Serverless (vector engine) |
| Hybrid keyword (BM25) + semantic search as a first-class capability | Amazon OpenSearch Serverless |
| Small corpus already living in Aurora PostgreSQL | Aurora pgvector |
| Fully managed RAG with minimal code and a small team | Amazon Bedrock Knowledge Bases (can be backed by S3 Vectors) |
| Graph relationships plus vector search | Amazon Neptune Analytics |
| Vector search over DynamoDB data | DynamoDB zero-ETL to OpenSearch |

Why S3 Vectors here: the workload is an internal knowledge assistant where a warm-query latency around 100 ms is invisible inside an end-to-end LLM response of several seconds, and storage cost dominates at corpus scale. S3 Vectors is priced dramatically below dedicated vector engines for exactly this shape of workload. It is the wrong choice for high-QPS realtime retrieval or hybrid keyword search; I would move to OpenSearch Serverless for either.

## Managed vs DIY

**Bedrock Knowledge Bases** handles ingestion, chunking, embedding and retrieval as a managed workflow, with a RetrieveAndGenerate API and built-in citations. Choose it for speed to production with a small team.

**DIY with boto3** (this repo) owns the ingestion pipeline in exchange for full control over chunking strategy, embedding model, reranking, prompt templates and evaluation. Choose it when retrieval quality requirements are specific enough that the managed defaults are the bottleneck, or when the team needs to understand and tune every stage.

This repo deliberately takes the DIY path because the goal is to expose and reason about every decision.

## Decisions encoded in the code

1. **Same embedding model at index and query time.** Enforced structurally: one `embed_text` function serves both paths, and dimension mismatches raise instead of degrading silently.
2. **Text stored as non-filterable metadata next to the vector.** The LLM never sees vectors; it needs text. Storing text with the vector makes retrieval a single call, and non-filterable metadata carries a higher size limit.
3. **Immutable index config treated as a versioning event.** Dimension, distance metric and non-filterable keys cannot change after index creation. Changing the embedding model therefore means a new index and a re-ingest, which is why every chunk records a `chunking_version` and uses deterministic IDs for idempotent re-ingestion.
4. **Retrieval-time security via server-side metadata filters.** `tenant_id` and `access_group` are attached at ingestion; the query layer injects filters like `{"tenant_id": {"$eq": "acme"}}` derived from the authenticated identity, never from client input.
5. **Distance-threshold refusal before generation.** When the nearest hit is too far from the question, the pipeline says "I don't know" without calling the LLM. Retrieval distance is the cheapest hallucination tripwire available.
6. **Provider-agnostic generation via the Converse API.** The generation model is a swappable, low-leverage component; the code reflects that.

## Production security checklist

- Customer-managed KMS keys on the vector bucket for regulated data
- VPC endpoints for `s3vectors` and `bedrock-runtime` so traffic stays off the public internet
- Least-privilege IAM with resource-specific ARNs (note: `returnMetadata=True` plus filters needs both `s3vectors:QueryVectors` and `s3vectors:GetVectors`)
- CloudTrail on every `put_vectors`, `query_vectors` and `invoke_model` call
- Server-side tenant filters on every query; audit metadata on every chunk
- Bedrock Guardrails for PII detection and content policy at the generation layer
