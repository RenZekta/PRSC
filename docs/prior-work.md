# Selected prior work (checked 15 November 2026)

This is a selective reading map, not an exhaustive literature or patent search.

| Work | Verified venue/status | PRSC relevance |
|---|---|---|
| Mu, Li & Goodman, *Learning to Compress Prompts with Gist Tokens* | NeurIPS 2023 | Learned compact activations that can replace repeated prompt context. |
| Wu et al., *Memorizing Transformers* | ICLR 2022 | Approximate nearest-neighbor memory over past internal key-value representations. |
| Eyuboglu et al., *Cartridges* | ICLR 2026 | Corpus-specific KV representations trained offline; especially close prior art. |
| Li et al., *End-to-End Context Compression at Scale* | arXiv preprint, 2026 | LCLMs compress context and support selective segment expansion. |
| Lu et al., *TurboRAG* | EMNLP 2025 | Precomputed chunk-level KV caches with inference-time retrieval/composition. |
| Liu et al., *CacheGen* | ACM SIGCOMM 2024 | Compressed KV-cache bitstreams and adaptive streaming. |
| Jiang et al., *LLMLingua* | EMNLP 2023 | Token/prompt compression baseline. |
| Behrouz et al., *Titans* | NeurIPS 2025 | Learned long-term memory complementing attention. |
| Li et al., *SCBench* | ICLR 2025 | Evaluates KV-cache generation, compression, retrieval, and loading. |
| Li, *Models Take Notes at Prefill* | arXiv preprint, 2026 | Editable/composable KV-cache notes; relevant but not peer-reviewed in the cited version. |

## Primary sources

- Gist Tokens: https://arxiv.org/abs/2304.08467
- Memorizing Transformers: https://arxiv.org/abs/2203.08913
- Cartridges: https://proceedings.iclr.cc/paper_files/paper/2026/hash/4681359a7b1e94571598ad1adda35e6e-Abstract-Conference.html
- LCLMs: https://arxiv.org/abs/2606.09659
- TurboRAG: https://aclanthology.org/2025.emnlp-main.334/
- CacheGen: https://doi.org/10.1145/3651890.3672274
- LLMLingua: https://aclanthology.org/2023.emnlp-main.825/
- Titans: https://proceedings.neurips.cc/paper_files/paper/2025/hash/a4ca07aa108036f80cbb5b82285fd4b1-Abstract-Conference.html
- SCBench: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a540b17fb2295c736d5afd6c507acf66-Abstract-Conference.html
- Models Take Notes: https://arxiv.org/abs/2606.17107

## Positioning caution

Cartridges, LCLMs, and composable/precomputed KV work make it especially important not to claim novelty for offline corpus memory, latent context compression, selective expansion, or cache reuse in isolation. PRSC's proposed distinction is the explicit combination of (a) a model-conditioned pattern catalogue, (b) residual-plus-reference K-gram records, (c) typed graph relations and provenance, and (d) a confidence-aware activation controller that can choose among compatible cache loading, reconstruction, and exact-source expansion. This combination is a hypothesis requiring a deeper novelty search and empirical validation.
