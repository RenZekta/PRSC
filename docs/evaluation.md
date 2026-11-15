# PRSC evaluation protocol

This document operationalizes the hypotheses in the paper. It is a plan, not a report of completed experiments.

## Primary unit of analysis

A workload item is a tuple `(archive, query, target, precision_requirement)`. Every method is evaluated on the same archive/query split and base model when technically feasible.

## Required methods

1. Full relevant context.
2. Conventional text RAG.
3. LLMLingua or an equivalent prompt compressor.
4. Gist/latent context method where available.
5. Cartridges-style corpus memory where available.
6. Precomputed KV retrieval (e.g. TurboRAG-style) and cache transfer compression (e.g. CacheGen-style) where applicable.
7. PRSC full system and ablations.

Record model differences when exact baseline parity is impossible.

## Minimum task families

- exact symbol and identifier lookup;
- single-file and cross-file code questions;
- dependency and change-impact analysis;
- debugging and patch generation with test execution;
- documentation QA and multi-document synthesis;
- temporal changes and contradictory versions;
- needle-in-haystack and global information tasks;
- adversarial minimal pairs where one token changes meaning.

## Metrics

Report quality and cost jointly:

- answer exact match/F1 and citation/provenance correctness;
- code test pass rate and patch acceptance;
- retrieval recall and dependency recall;
- source-to-archive bytes, active tokens, active KV bytes;
- index latency, compile cost, TTFT, decode throughput, end-to-end latency;
- GPU/CPU memory and network bytes;
- query count to amortize preprocessing and caches;
- confidence calibration and silent omission rate.

Use Pareto curves and workload-frequency sensitivity analysis. Include cold-cache and warm-cache results.

## Mandatory stress tests

- negation and changed numeric constants;
- rare identifiers and names;
- near-identical code with different behavior;
- stale package/API versions;
- missing or contradictory dependencies;
- source edits, deletion, and transitive invalidation;
- base model/tokenizer revision changes;
- malformed or poisoned source content;
- cache mismatch and corrupted cache payload;
- broad queries requiring many or all archive records.

## Reporting discipline

Publish exact versions, hashes, configurations, prompts, retrieval budgets, and baseline adaptations. Do not report compression ratios without task quality and end-to-end latency. Do not infer theoretical logarithmic behavior from a small empirical curve. Report uncertainty intervals over seeds and query sets.
