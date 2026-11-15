# Prior-Relative Sparse Context (PRSC)

**Knowledge Grams for Selective Activation of Model-Relative Memory**

PRSC is a research proposal for an LLM context architecture in which an external archive is compiled into addressable **Knowledge Grams (K-grams)**. A gram references a reusable pattern, stores the instance-specific residual, records provenance and dependencies, and may point to a compatible precomputed KV representation. At query time, a controller activates only the relevant working set and expands exact source when needed.

> **Status: proposal / research specification.** This repository does not contain a trained encoder, production retrieval engine, or experimental results. The paper states hypotheses and an evaluation plan; it does not claim that PRSC has been empirically validated.

## Start here

- [Research paper (PDF)](paper/paper.pdf)
- [Research paper (Markdown source)](paper/paper.md)
- [LaTeX source for submission workflows](paper/paper.tex)
- [Architecture and implementation notes](docs/architecture.md)
- [Selected prior-work map](docs/prior-work.md)
- [Evaluation protocol](docs/evaluation.md)
- [K-gram JSON Schema](schemas/knowledge-gram.schema.json)
- [Example K-gram](examples/knowledge-gram.example.json)

## Core idea

A conventional long-context pipeline repeatedly turns a large body of source material into active model context. PRSC instead factors each source unit into:

1. a reference to a reusable, versioned pattern in a model-conditioned catalogue;
2. the residual information that distinguishes this instance;
3. typed relationships and provenance;
4. an optional pointer to a validated precomputed representation.

The query controller retrieves relevant grams, expands dependencies, checks confidence and cache compatibility, and chooses between precomputed KV activation, reconstruction from pattern plus residual, or exact-source expansion.

The phrase **prior-relative** does not imply that neural weights expose stable addresses for concepts. The proposed pattern catalogue is an explicit interface that must be built, versioned, and validated.

## Repository layout

```text
paper/       Proposal paper, figure, and PDF
schemas/     JSON Schema for K-grams
examples/    Example K-gram record
docs/        Architecture and evaluation notes
src/prsc/    Small standard-library K-gram validator
tests/       Unit tests for the validator
```

## Validate the example

Python 3.10+; no third-party dependencies required.

```bash
python -m unittest discover -s tests -v
python -m prsc.validate examples/knowledge-gram.example.json
```

Regenerate the PDF and standalone LaTeX source with Pandoc and XeLaTeX installed:

```bash
make paper
```

To run the module directly from the repository root:

```bash
PYTHONPATH=src python -m prsc.validate examples/knowledge-gram.example.json
```

Regenerate the PDF and standalone LaTeX source with Pandoc and XeLaTeX installed:

```bash
make paper
```

The validator checks a small set of invariants. The JSON Schema is the normative structural draft; the validator is intentionally not a cache compatibility checker or a semantic correctness proof.

## Key design constraints

- Never discard source-critical details solely because they are predictable or low-surprisal.
- Keep provenance, hashes, versioning, and exact-source fallback.
- Treat retrieved source as untrusted data, not as instructions.
- Reject incompatible KV caches; matching tensor shape is not enough.
- Measure compile cost, storage, network transfer, activation latency, and task quality together.
- Do not describe the system as having logarithmic total context cost. The target is sparse active context with sublinear or logarithmic lookup in suitable indexes.

## Related work

The most relevant precedents are Cartridges (corpus-specific KV representations), LCLM (latent context with on-demand expansion), TurboRAG (precomputed chunk-level KV retrieval), CacheGen (KV compression and streaming), Gist Tokens, Memorizing Transformers, Titans, and SCBench. Full references and the proposed distinction are in the paper.

## Publication checklist

- [ ] Replace the author placeholder and add affiliations/contact details.
- [ ] Run a fresh literature and patent/prior-art search before claiming novelty.
- [ ] Add experimental results only after reproducing baselines and recording full costs.
- [ ] Verify licenses for any included data, code, models, or derived artifacts.
- [ ] Update the date/version, bibliography, and `CITATION.cff`.

## License

The paper is licensed under [CC BY 4.0](LICENSE-PAPER). The software and schema are licensed under the MIT License (see `LICENSE`).
