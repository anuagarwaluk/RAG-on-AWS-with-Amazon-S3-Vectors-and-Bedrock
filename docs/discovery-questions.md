# Discovery Before Architecture

The most expensive RAG mistake is architecting before qualifying. When a stakeholder says "we want to train an LLM on our data," in most cases the actual requirement is RAG, and the way to establish that is a structured discovery conversation, not a proof of concept. These are the questions I run first, and what each answer changes.

1. **What business problem is this solving?** Ticket deflection, employee productivity, compliance lookup, customer self-service. If nobody can articulate this, the project has no success metric and should not proceed to architecture.

2. **Who are the users and how many?** Fifty internal analysts querying a few times a day points at S3 Vectors. Millions of external customers at peak hours points at OpenSearch Serverless. QPS drives the vector store decision more than corpus size does.

3. **What is the source data?** Volume (GB or TB), variety (PDFs, wikis, tickets, code, images), velocity (annual refresh or hourly churn), sensitivity (public, confidential, PII, regulated). Velocity determines the ingestion architecture; sensitivity determines the security architecture.

4. **What is the answer quality bar?** "Good enough to draft a first response" and "defensible in a compliance audit" are different systems with different evaluation, thresholds and human-review requirements.

5. **What happens when the system does not know?** The correct behaviour is an explicit "I don't know," which requires distance thresholds and grounded prompting from day one, not as a retrofit.

6. **Who is allowed to see what?** If different users may retrieve different documents, access control must live in the retrieval layer (metadata filters enforced server side), not in the UI.

7. **What is the freshness requirement?** Retrieval answers are only as current as the last ingestion. Hourly-changing data needs event-driven ingestion; a static handbook does not.

8. **Is there an existing search or data platform to meet where it is?** An organisation already on Aurora PostgreSQL may be better served by pgvector than by a new service.

9. **What team will own this in production?** No ML team and a tight timeline favours Bedrock Knowledge Bases; a platform team with specific quality requirements favours the DIY control demonstrated in this repo.

10. **How will we measure success?** A labelled evaluation set (question, expected source) must exist before tuning starts. Without it, every chunking or top-k change is guesswork. See `src/rag_pipeline/evaluate.py`.
