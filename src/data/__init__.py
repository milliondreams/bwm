"""BWM data pipeline — point-in-time multimodal ingest.

Layout mirrors the spec (§5.6):
- schemas/   canonical Pydantic records (PIT envelope, entity, financial fact, …)
- storage/   fsspec-based backend (local + AzureML datastore blob)
- pit/       point-in-time engine (as-of queries, restatement-aware)
- entity/    entity resolution + registry
- sources/   one module per data source (edgar, fred, market, …)
- graph/     graph construction
- cli/       command-line entry points for ingest
"""
