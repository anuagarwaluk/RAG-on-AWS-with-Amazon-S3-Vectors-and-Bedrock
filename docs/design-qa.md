# Design Q&A

Questions I expect from a technical reviewer of this system, answered the way I would answer them in a design review.

**Why store the raw text in the vector database at all?**
The LLM never receives vectors; vectors only locate the relevant chunks. Once located, the actual text must be sent to the model, so it has to live somewhere. Storing it beside the vector makes retrieval a single API call returning both match and content. A separate text store keyed by chunk ID adds a second lookup and more failure modes for no meaningful saving. In S3 Vectors the text field is marked non-filterable, which raises its size limit and keeps it out of the filter index.

**A user reports the assistant gives wrong answers. Where do you look first?**
Not at the LLM. In order: (1) is the right chunk being retrieved at all (inspect top-k results and distances), (2) is chunking splitting the answer across chunk boundaries, (3) is the embedding model aligned with the domain vocabulary, (4) is top-k drowning the answer in weakly relevant context, and only then (5) the prompt and the model. In practice the failure is in retrieval far more often than in generation.

**How do you make the system say "I don't know" instead of hallucinating?**
Two layers. The prompt instructs the model to answer only from context and to refuse otherwise. Before that, the pipeline checks the nearest-hit distance against a tuned threshold and refuses without calling the LLM when even the best match is weak. The threshold is set empirically by comparing distance bands for answerable versus unanswerable questions (see `evaluate.no_answer_distance_probe`).

**How does multi-tenant isolation work when all tenants share one index?**
Every chunk carries a `tenant_id` in filterable metadata at ingestion. At query time the application backend derives the tenant from the authenticated token and injects `{"tenant_id": {"$eq": ...}}` into every `query_vectors` call. The filter is added server side only; accepting it from the client would let a user spoof another tenant. Index-per-tenant remains the right call for very large tenants needing performance isolation.

**When is S3 Vectors the wrong choice?**
Single-digit-millisecond latency at high QPS, realtime updates that must be queryable in under a second, or first-class hybrid BM25-plus-vector search: all of those point at OpenSearch Serverless. A small corpus already living in Aurora points at pgvector. The cost advantage of S3 Vectors only matters when the workload tolerates its latency profile.

**Fine-tuning vs RAG in one breath?**
Fine-tuning changes the model's weights; RAG keeps the model frozen and supplies knowledge at query time. RAG updates by editing documents, cites sources naturally, keeps sensitive data out of model weights, and reaches production in days rather than weeks. Fine-tuning earns its cost when the requirement is style, format, or behaviour, not knowledge.

**What are the highest-leverage quality levers, in order?**
Chunking strategy and size, embedding model alignment with the domain, reranking, and only then the choice of LLM. Stakeholders usually assume the opposite order.
