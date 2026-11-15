# PRSC architecture notes

## Design objective

Represent a large external archive in a persistent structure whose *active* model context is selected per query. The archive is compiled relative to a specified base model and pattern catalogue, but remains external to the model weights.

## K-gram contract

A K-gram is a record with:

- a stable archive-local identifier;
- a compatibility declaration for the base model, tokenizer, encoder, and adapters;
- a typed pattern reference (or `null` when no safe pattern exists);
- a residual payload and confidence estimate;
- typed links to related grams, including dependencies that must be activated together;
- one or more provenance records with source hashes and spans;
- an activation policy and optional KV manifest;
- an exact-source fallback.

A K-gram is not defined as a single token, n-gram, or KV tensor. It may be symbolic, textual, latent, or a reference to a cached neural state.

## Suggested pattern-catalogue entry

```json
{
  "pattern_id": "PATTERN_ASYNC_DB_LOOKUP_V1",
  "modality": "code",
  "type": "semantic_template",
  "schema_version": "1.0",
  "model_fingerprint": "sha256:...",
  "template": "async_database_lookup_with_error_handling",
  "allowed_tasks": ["code_qa", "dependency_analysis"],
  "validation_suite": "pattern-tests-001"
}
```

For code, a first catalogue can use AST templates, function signatures, symbol types, and dependency motifs. For prose, it may use entity-relation frames, document schemas, or learned latent prototypes. The catalogue must not be treated as ground truth: a match is a proposal that must pass validation.

## Activation policy

Use a risk-aware three-tier policy:

1. **Exact:** source bytes/tokens or lossless structure for exact retrieval, code generation, sensitive decisions, and audit.
2. **Validated:** pattern plus residual only after reconstruction/invariant tests pass.
3. **Approximate:** latent/cache representation for workloads where measured task-level distortion is acceptable.

Any low-confidence match, conflicting version, missing provenance, or cache fingerprint mismatch escalates to a safer tier.

## KV-cache manifest requirements

A cache manifest should include model weights/revision, tokenizer, adapter stack, attention and positional encoding configuration, layer/head layout, dtype/quantization, serialization format, position semantics, and the canonical conditioning context. A production implementation also needs cryptographic hashes and a reproducible cache-generation procedure. Cache reuse must be tested against full prefill, including boundary and cross-chunk behavior.

## Update and deletion semantics

Updates create a new archive version. Changed source spans invalidate dependent grams and caches. Deletion must remove or rebuild derived records and indexes, not merely hide the original source. A dependency graph supports transitive invalidation.
