# My annotated bibliography — RAG for university admission regulations

**Thesis:** An Empirical Evaluation of RAG Pipeline Components and Production Trade-Offs for University Admission Regulations

**Updated:** 2026-11-23

**Corpus:** 23 PDFs in `papers/pdf/` plus 2 web-only sources (BGE reranker card, Cohere docs)

**BibTeX:** `papers/references.bib` (25 entries)

## Legend

| Verdict | Meaning |
|---|---|
| **Core** | Directly informs an experiment, or is one of the closest prior systems. |
| **Supporting** | Framing, related work, or a component justification. |
| **Marginal** | Optional. Only usable under a specific design choice. |

## RQ mapping

| Tag | Scope |
|---|---|
| **RQ1** | Representation layer: chunking, parent-child, metadata injection. |
| **RQ2** | Query reasoning engines: HyDE, AutoRetriever, SubQuestion, Router, FLARE, rerank. |
| **RQ3** | Conversational multi-turn handling: rewrite vs carry-forward. |
| **EVAL** | Dataset construction and metrics. |
| **PROD** | Latency, throughput, cost, Pareto analysis. |

## A. Closest prior systems

### P01 · `elchafei2026hrag`

**H-RAG at SemEval-2026 Task 8: Hierarchical Parent-Child Retrieval for Multi-Turn RAG Conversations**

- **File:** [`papers/pdf/2026.semeval-1.155.pdf`](<papers/pdf/2026.semeval-1.155.pdf>)
- **Authors:** Passant Elchafei, Hossam Emam, Mohamed Alansary, Monorama Swain, Markus Schedl
- **Venue:** Proceedings of the 20th International Workshop on Semantic Evaluation (SemEval-2026), pp. 1136-1143, San Diego, USA, ACL
- **Verdict:** CORE
- **RQ:** RQ1 (parent-child), RQ3 (multi-turn)
- **DOI:** [10.18653/v1/2026.semeval-1.155](https://doi.org/10.18653/v1/2026.semeval-1.155)

**Method.** Hierarchical parent-child RAG. Documents are split into overlapping sentence-based child chunks for retrieval. Full documents are kept as parent units for generation. Child retrieval is followed by parent-level evidence aggregation and an instruction-tuned LLM. Evaluated on MTRAGEval Task A (standalone retrieval) and Task C (end-to-end multi-turn generation with retrieved evidence).

**Key findings.** nDCG@5 = 0.4271 on Task A. Task C harmonic mean = 0.3241 (RBagg 0.2488, RLF 0.2703, RBllm 0.6508). The authors attribute much of the result to retrieval configuration and parent-level aggregation.

**Use for my work.** Direct methodological precedent for my RQ1 parent-child arm and my RQ3 retrieval carry-forward comparison. It is also evidence that returning coherent parent context (rather than isolated child chunks) matters for multi-turn faithfulness.

**Caveats.** Competition task, not admission regulations, with the full document as the parent unit rather than a section/article. No production latency/cost analysis, and the reported scores are system-specific rather than a general benchmark of parent-child methods.

**Placement in my thesis.** Related work (RQ1, RQ3) and design justification for parent-child retrieval and for comparing carry-forward vs rewriting.

### P02 · `nguyen2025urag`

**URAG: Implementing a Unified Hybrid RAG for Precise Answers in University Admission Chatbots - A Case Study at HCMUT**

- **File:** [`papers/pdf/2501.16276v1.pdf`](<papers/pdf/2501.16276v1.pdf>)
- **Authors:** Long S. T. Nguyen, Tho T. Quan
- **Venue:** Information and Communication Technology (SOICT 2024), Communications in Computer and Information Science, pp. 82-93, Springer, 2025
- **Verdict:** CORE (domain precedent)
- **RQ:** RQ2 (hybrid retrieval), PROD (cost/complexity)
- **DOI:** [10.1007/978-981-96-4285-4_7](https://doi.org/10.1007/978-981-96-4285-4_7)

**Method.** Unified hybrid RAG for a university admission chatbot. Two-tier rule-based + RAG design. Two enrichment mechanisms: URAG-D (document database augmentation) and URAG-F (enhanced FAQ generation). Targets a lightweight in-house LLM to avoid the cost of heavy iterative RAG. Evaluation on 500 real high-school questions, with expert correctness assessment.

**Key findings.** URAG raises the in-house lightweight model to performance comparable with state-of-the-art commercial models. The case study at HCMUT reports positive user feedback. The paper explicitly frames advanced RAG (iterative feedback, specialized modules) as too costly and complex for practical deployment.

**Use for my work.** The closest admission-domain architecture. I cite it for the hybrid retrieval + FAQ/document augmentation rationale and for the cost constraint that motivates my production analysis. Useful contrast: URAG optimizes a lightweight LLM, whereas I hold the LLM fixed and compare retrieval/query-engine architectures.

**Caveats.** No component-wise ablation of chunking or query engines and no public benchmark. The study is Vietnamese in institution and language, and the hybrid retrieval internals are not described at the depth needed to reproduce them.

**Placement in my thesis.** Related work (admissions domain) and PROD motivation.

### P03 · `khasanova2025design`

**Design and Performance Evaluation of LLM-Based RAG Pipelines for Chatbot Services in International Student Admissions**

- **File:** [`papers/pdf/electronics-14-03095.pdf`](<papers/pdf/electronics-14-03095.pdf>)
- **Authors:** Maksuda Khasanova Zafar kizi, Youngjung Suh
- **Venue:** Electronics, 14(15):3095, MDPI, 2025
- **Verdict:** CORE — closest prior experimental design
- **RQ:** RQ1, RQ2, PROD
- **DOI:** [10.3390/electronics14153095](https://doi.org/10.3390/electronics14153095)

**Method.** Domain-specific admission chatbot for international students. Compares retrieval methods (Dense, MMR, Hybrid), chunking strategies (Semantic, Recursive), and open-source vs commercial LLMs (GPT-4o, LLaMA3, OpenChat). Uses two evaluation sets (LLM-generated and human-tagged QA) and RAGAS metrics (answer relevancy, faithfulness, context precision, context recall) plus heuristic NLP metrics. Also measures latency across RAG stages.

**Key findings.** OpenChat 7b with recursive splitting and hybrid retrieval achieved the highest average scores on both datasets and outperformed GPT-4o on key RAGAS metrics. Open-source pipelines can be comparable and cost-efficient. Latency was the main cost: large models (e.g., a 14 GB OpenChat) increased retrieval/generation time, and the authors flag a <3-5 s target for real-time chat.

**Use for my work.** The primary system I position against. My delta is the combination it does not test: Docling-native AST hierarchical chunking + fixed-size parent-child + metadata injection, BGE-M3 dense + sparse, the query-engine suite (HyDE, Metadata AutoRetriever, SubQuestion, Router, FLARE, cross-encoder rerank), RQ3 multi-turn, and a Pareto frontier with throughput and $/1k queries, not only latency. I cite it in my introduction and in a dedicated contrast paragraph.

**Caveats.** No query-engine comparison, no multi-turn handling, no Docling AST or metadata injection, and no throughput or cost-per-query Pareto. Chunking is semantic/recursive only, and the study is single-institution.

**Placement in my thesis.** Related work (closest system), novelty statement, and RQ1/RQ2/PROD.

### P04 · `setiawan2026performance`

**Performance Analysis of Reasoning Models in RAG-Based Question Answering System for University Admission Services**

- **File:** [`papers/pdf/3707+-+Surya+3789-3798.pdf`](<papers/pdf/3707+-+Surya+3789-3798.pdf>)
- **Authors:** Muhammad Surya Adhi Setiawan, Arista Pratama, Prasasti Karunia Farista Ananto
- **Venue:** Bit-Tech (Binary Digital - Technology), 8(3):3789-3798, 2026
- **Verdict:** CORE/SUPPORTING (domain + RQ2 reasoning models)
- **RQ:** RQ2, EVAL
- **DOI:** [10.32877/bt.v8i3.3707](https://doi.org/10.32877/bt.v8i3.3707)

**Method.** RAG QA for university admissions built from 30 official admission sources with VoyageAI embeddings. 353 real multilingual queries (Indonesian, English, Javanese). Retrieval and prompts are fixed across models. Compares reasoning models (DeepSeek-R1, Gemini-2.5-Flash, o4-mini) with non-reasoning models (DeepSeek-V3, Gemini-2.0-Flash, GPT-4o-mini) using six RAGAS metrics.

**Key findings.** Reasoning models average 0.7772 vs 0.7289 for non-reasoning (+6.63%). The largest gain is factual correctness (+15.95%). Gemini-2.5-Flash has the best composite score (0.8207) with favorable cost. Reasoning models are more stable across languages. The authors stress that overall performance still depends strongly on retrieval quality.

**Use for my work.** Domain precedent for admission RAG. Supports my RQ2 discussion of reasoning vs non-reasoning engines and the multilingual robustness claim. I cite it when explaining why I hold the generator fixed (or when discussing reasoning-model choice) so that retrieval effects are not confounded with LLM effects.

**Caveats.** Compares LLMs, not retrieval architectures, with a fixed retriever and no production throughput/Pareto analysis. The language setting is Indonesian/Javanese rather than Romanian, and the dataset is single-institution.

**Placement in my thesis.** Related work (domain, RQ2) and discussion of model choice.

### P17 · `wojtasik2025polichat`

**PoliChat: Retrieval Augmented Generation on University Documents and Regulations**

- **File:** [`papers/pdf/wojtasik2025polichat.pdf`](<papers/pdf/wojtasik2025polichat.pdf>)
- **Authors:** Konrad Wojtasik, Adrian Berdowski, Inez Okulska, Maciej Piasecki
- **Venue:** Computational Science - ICCS 2025 Workshops, Lecture Notes in Computer Science, pp. 273-288, Springer, 2025
- **Verdict:** CORE (closest same-domain system)
- **RQ:** RQ1 (document length, summarization, context size), RQ2 (retrieval strategies, reranking), EVAL (annotated dataset), PROD (model size)
- **DOI:** [10.1007/978-3-031-97570-7_21](https://doi.org/10.1007/978-3-031-97570-7_21)

**Method.** RAG chatbot for university regulations at Wrocław University of Science and Technology. Real-time retrieval with explicit source citations. Uses bge-m3 and a bge reranker. Builds a 100-question manually annotated evaluation set for IR (relevant chunks) and generation (answers and citations). Varies model size, document length, retrieval strategies, summarization, retrieved-context size, prompting, and citation handling. Introduces Analyze & Answer (A&A) prompting.

**Key findings.** Larger models perform better, but retrieval quality remains critical. Longer documents reduce recall. Extractive summaries retain more relevant information than abstractive ones. Citations improve transparency but can reduce correctness, and longer retrieved contexts make citation selection harder. Analyze & Answer improves overall performance, citation accuracy, and readability.

**Use for my work.** The closest domain match (university regulations rather than admissions specifically) and direct evidence for my RQ1/RQ2 choices: bge-m3 + reranker is a defensible domain baseline, document length and context size matter, and citation behavior is a real trade-off. Its annotated dataset is a model for my own synthetic-plus-annotated evaluation.

**Caveats.** Polish regulations, not Romanian admission documents. The paper studies document length, summarization, context size, prompting, and citations, not the retriever/query-engine matrix in my RQ2. It has no production throughput/cost Pareto analysis. The 100-question set is small.

**Placement in my thesis.** Related work (closest domain system), RQ1 context-size and summarization decisions, and EVAL dataset design.

## B. RQ1 — Representation layer: chunking, parent-child, metadata

### P05 · `lu2026hichunk`

**HiChunk: Evaluating and Enhancing Retrieval Augmented Generation with Hierarchical Chunking**

- **File:** [`papers/pdf/2509.11552v3.pdf`](<papers/pdf/2509.11552v3.pdf>)
- **Authors:** Wensheng Lu, Keyu Chen, Zhifeng Shen, Ruizhi Qiao, Xing Sun
- **Venue:** Proceedings of the 64th Annual Meeting of the ACL (Volume 1: Long Papers), pp. 29738-29753, 2026
- **Verdict:** CORE
- **RQ:** RQ1, EVAL
- **DOI:** [10.18653/v1/2026.acl-long.1372](https://doi.org/10.18653/v1/2026.acl-long.1372)

**Method.** Introduces HiCBench: manually annotated multi-level document chunking points, synthesized evidence-dense QA pairs, and evidence sources. Proposes HiChunk, a multi-level document structuring framework built with fine-tuned LLMs, combined with Auto-Merge retrieval.

**Key findings.** Existing RAG benchmarks are inadequate for evaluating chunking because of evidence sparsity. HiCBench discriminates the effect of different chunking methods across the whole pipeline. HiChunk + Auto-Merge improves chunking quality, retrieval, and response quality within reasonable time.

**Use for my work.** The strongest justification for multi-level hierarchical chunks plus Auto-Merge/parent-child retrieval in RQ1. HiCBench's evidence- sparsity argument supports chunk-level retrieval evaluation, which I operationalize with chunkID matching.

**Caveats.** Uses fine-tuned LLMs to build the hierarchy, which is expensive, whereas my Docling AST approach is cheaper and domain-structural. The corpus is not admission regulations, and reported gains are relative to the paper's own baselines.

**Placement in my thesis.** RQ1 design and evaluation methodology (evidence sparsity, multi-level chunking).

### P06 · `taiwo2026evaluating`

**Evaluating Chunking Strategies for Retrieval-Augmented Generation in Oil and Gas Enterprise Documents**

- **File:** [`papers/pdf/2603.24556v1.pdf`](<papers/pdf/2603.24556v1.pdf>)
- **Authors:** Samuel Taiwo, Mohd Amaluddin Yusoff
- **Venue:** Computer Science and Information Technology (CS and IT), pp. 49-67, CCSEIT 2026
- **Verdict:** CORE
- **RQ:** RQ1, PROD
- **DOI:** [10.5121/csit.2026.160507](https://doi.org/10.5121/csit.2026.160507)

**Method.** Empirical comparison of four chunking strategies — fixed-size sliding window, recursive, breakpoint-based semantic, and structure-aware — on a proprietary oil-and-gas corpus containing text-heavy manuals, table-heavy specifications, and piping-and-instrumentation diagrams (P&IDs). Measures retrieval effectiveness and computational cost.

**Key findings.** Structure-aware chunking has the highest overall retrieval effectiveness, especially in top-K metrics, and incurs significantly lower computational cost than semantic or baseline strategies. All four text-only methods perform poorly on visually/spatially encoded P&IDs.

**Use for my work.** Supports my choice of Docling structure-aware hierarchical chunking over fixed-size and semantic chunking, including the cost argument. The P&ID failure is a direct caution for admission PDFs whose tables carry grade cutoffs, fees, and deadlines: chunking cannot rescue bad table extraction, which is why the Docling parsing layer matters.

**Caveats.** Proprietary single-domain corpus, retrieval-focused rather than end-to-end answer quality, with "structure-aware" implemented differently from Docling's AST hierarchy. No Romanian or legal/regulatory text.

**Placement in my thesis.** RQ1 hypotheses, PROD cost argument, and parser/chunker rationale.

### P07 · `yang2026structure`

**Structure-Aware Semantic Chunking with Title-Chain Prefixes: A 1600-Query Evaluation and the Measurement Trap in Text-Transform Ablations**

- **File:** [`papers/pdf/2608.00824v1.pdf`](<papers/pdf/2608.00824v1.pdf>)
- **Authors:** Yang Yang
- **Venue:** arXiv:2608.00824, 2026
- **Verdict:** SUPPORTING/CORE (metadata injection, evaluation caution)
- **RQ:** RQ1, EVAL
- **DOI:** [10.48550/arXiv.2608.00824](https://doi.org/10.48550/arXiv.2608.00824)

**Method.** Three-stage chunk-side pipeline: header split, semantic merge, and title-chain prefixing. The title chain reuses the document's own header hierarchy and costs zero additional LLM calls. Evaluated on 1,600 stratified queries over a production Markdown knowledge base with two annotators.

**Key findings.** MRR@5 improves from 0.374 to 0.463 (+23.8%) on the full set and from 0.828 to 0.925 (+11.7%) on the answerable subset (n=563). In a same-pool prefix on/off ablation, inter-annotator agreement collapses from kappa 0.45 to 0.04, because stripping the title chain removes the disambiguation signal. The author calls this a measurement trap and recommends retrieval-time per-candidate prefix evaluation.

**Use for my work.** Direct support for heading-chain (title-chain) injection in RQ1. It also matters for my current implementation: I carry heading paths as metadata and only repair orphan headings from empty sections, whereas Yang prefixes every chunk inline. The result supports the value of the chain, not the specific carrier. I use the measurement-trap warning when I ablate enrichment: I should not strip the context while annotating relevance.

**Caveats.** Single-author cs.SE paper with one Markdown KB, not admission regulations, and no Docling AST comparison. The effect size may not transfer to Romanian legal-style text. The "measurement trap" concerns annotation agreement, not retrieval scores directly.

**Placement in my thesis.** RQ1 metadata/heading injection and the threats-to-validity section.

### P08 · `wasson2002using`

**Using Summaries in Document Retrieval**

- **File:** [`papers/pdf/W02-0405.pdf`](<papers/pdf/W02-0405.pdf>)
- **Authors:** Mark Wasson
- **Venue:** Proceedings of the Workshop on Automatic Summarization (including DUC 2002), pp. 37-44, ACL, 2002
- **URL:** <https://aclanthology.org/W02-0405/>
- **Verdict:** MARGINAL
- **RQ:** RQ1 (summary-based metadata, optional)

**Method.** Thirty searches against full text vs summaries only in large news collections. Two evaluation scopes.

**Key findings.** Summary-only retrieval benefits users who want small answer sets focused on highly relevant documents. Users who need all references to a topic should search full text.

**Use for my work.** Only if I add summary fields as metadata/payload injection. It is historical evidence that summaries trade recall for precision, which is a useful caveat for the RQ1 metadata augmentation arm.

**Caveats.** 2002 news retrieval, no RAG or LLMs, and not directly comparable to modern chunk enrichment. Citation is optional.

**Placement in my thesis.** Optional related work on summary-based enrichment.

## C. RQ2 — Query reasoning and retrieval engines

### P09 · `gao2023precise`

**Precise Zero-Shot Dense Retrieval without Relevance Labels**

- **File:** [`papers/pdf/2212.10496v1.pdf`](<papers/pdf/2212.10496v1.pdf>)
- **Authors:** Luyu Gao, Xueguang Ma, Jimmy Lin, Jamie Callan
- **Venue:** Proceedings of the 61st Annual Meeting of the ACL (Volume 1: Long Papers), pp. 1762-1777, 2023
- **Verdict:** CORE
- **RQ:** RQ2 (HyDE arm)
- **DOI:** [10.18653/v1/2023.acl-long.99](https://doi.org/10.18653/v1/2023.acl-long.99)

**Method.** HyDE. An instruction-following LLM generates a hypothetical document for the query. An unsupervised contrastive encoder embeds that document. Retrieval uses the generated vector. No relevance labels are needed.

**Key findings.** Improves zero-shot dense retrieval across web search, SciFact, ArguAna, TREC-COVID, FiQA, DBPedia-Entity, TREC-News, and Mr. TyDi. Captures relevance patterns without being factually correct. Works across languages with the appropriate prompts.

**Use for my work.** The RQ2 HyDE arm. Hypothesis: HyDE helps when student phrasing diverges from regulation vocabulary (informal/paraphrased queries) but may hurt exact-figure queries. Akarsu et al. 2026 found limited HyDE benefit for precise numerical questions, so I test both query profiles and report per-complexity results.

**Caveats.** Generation adds latency and cost, hallucinated details can mislead retrieval, and performance depends on the generator and prompt. Not tested on structured tables or Romanian.

**Placement in my thesis.** RQ2 design and hypotheses.

### P10 · `wang2024searching`

**Searching for Best Practices in Retrieval-Augmented Generation**

- **File:** [`papers/pdf/2407.01219v1.pdf`](<papers/pdf/2407.01219v1.pdf>)
- **Authors:** Xiaohua Wang, Zhenghua Wang, Xuan Gao, Feiran Zhang, Yixin Wu, Zhibo Xu, Tianyuan Shi, Zhengyuan Wang, Shizheng Li, Qi Qian, Ruicheng Yin, Changze Lv, Xiaoqing Zheng, Xuanjing Huang
- **Venue:** Proceedings of EMNLP 2024, pp. 17716-17736, Miami, USA, ACL
- **Verdict:** CORE (methodology)
- **RQ:** RQ1, RQ2, PROD
- **DOI:** [10.18653/v1/2024.emnlp-main.981](https://doi.org/10.18653/v1/2024.emnlp-main.981)

**Method.** Extensive component-wise ablation of a RAG workflow: chunking, retrieval (sparse/dense/hybrid), reranking, repacking, summarization, and generation, with attention to both effectiveness and efficiency. Also studies multimodal retrieval.

**Key findings.** No single module is universally optimal. The paper recommends combinations that balance performance and latency, and shows that multimodal retrieval helps visual inputs.

**Use for my work.** A template for my component-wise benchmark and for justifying a performance/efficiency lens rather than "add every advanced module." I use it to frame the production trade-off analysis.

**Caveats.** English QA benchmarks, not admissions or Romanian, and the modules do not map one-to-one onto my LlamaIndex query engines.

**Placement in my thesis.** Methodology, related work, and PROD framing.

### P11 · `akarsu2026bm25`

**From BM25 to Corrective RAG: Benchmarking Retrieval Strategies for Text-and-Table Documents**

- **File:** [`papers/pdf/2604.01733v1.pdf`](<papers/pdf/2604.01733v1.pdf>)
- **Authors:** Meftun Akarsu, Recep Kaan Karaman, Christopher Mierbach
- **Venue:** arXiv:2604.01733, 2026
- **Verdict:** CORE
- **RQ:** RQ2, EVAL, PROD
- **DOI:** [10.48550/arXiv.2604.01733](https://doi.org/10.48550/arXiv.2604.01733)

**Method.** Benchmarks ten retrieval strategies — sparse, dense, hybrid fusion (RRF and convex combination), cross-encoder reranking, query expansion, index augmentation, contextual retrieval, and adaptive/CRAG — on 23,088 queries over 7,318 text-and-table financial documents. Metrics: Recall@k, MRR, nDCG, and end-to-end Number Match, with paired bootstrap significance testing.

**Key findings.** (1) Hybrid retrieval + neural reranking is best: Recall@5 0.816, MRR@3 0.605, beating all single-stage methods by a large margin. (2) BM25 outperforms state-of-the-art dense retrieval on financial documents, challenging the assumption that semantic search always wins. (3) HyDE, multi-query, and adaptive retrieval give limited benefit for precise numerical queries, while contextual retrieval gives consistent gains.

**Use for my work.** The closest benchmark to my RQ2 and to admission documents full of exact figures (grade cutoffs, fees, deadlines). It directly motivates: (a) a strong BM25/sparse baseline, (b) hybrid + rerank as the expected winner, (c) testing HyDE separately on informal vs numeric queries, and (d) reporting paired significance tests rather than point estimates.

**Caveats.** Financial domain with OpenAI embeddings and Cohere rerank, English only, no multi-turn, no multilingual/Romanian, and a document mix that differs from regulations.

**Placement in my thesis.** RQ2 core, evaluation design, and PROD cost-accuracy discussion.

### P12 · `yan2026corrective`

**Corrective Retrieval Augmented Generation**

- **File:** [`papers/pdf/Corrective Retrieval Augmented Generation.pdf`](<papers/pdf/Corrective Retrieval Augmented Generation.pdf>)
- **Authors:** Shi-Qi Yan, Jia-Chen Gu, Yun Zhu, Zhen-Hua Ling
- **Venue:** Findings of the Association for Computational Linguistics: IJCNLP-AACL 2026, pp. 2991-3003, Hengqin, China, ACL
- **Verdict:** CORE/SUPPORTING (adaptive/corrective retrieval)
- **RQ:** RQ2 (router/multi-step), PROD
- **DOI:** [10.18653/2026.findings-aacl.151](https://doi.org/10.18653/2026.findings-aacl.151)

**Method.** CRAG. A lightweight T5-based retrieval evaluator assigns a confidence score to retrieved documents and triggers corrective actions (correct / ambiguous / incorrect), including web-search supplementation and decompose-then-recompose. The published version frames it as an autonomous agent that selects tools in a closed loop. Plug-and-play: no changes to the underlying RAG system.

**Key findings.** Outperforms prior RAG approaches on five datasets, including PopQA, Biography, PubHealth, and ARC-Challenge. The agent framing improves tool-calling efficiency and context quality.

**Use for my work.** RQ2 architecture-level comparison and justification for a corrective/adaptive engine or Router. If I do not implement CRAG, I cite it as the upper-bound adaptive approach and discuss its latency/cost.

**Caveats.** Original design depends on web search (not available/desirable for admission regulations), requires evaluator training, is not admission-domain, and adds latency. The published version differs from the 2024 arXiv version.

**Placement in my thesis.** RQ2 related work and router/corrective design.

### P13 · `jeong2024adaptiverag`

**Adaptive-RAG: Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity**

- **File:** [`papers/pdf/Adaptive-RAG Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.pdf`](<papers/pdf/Adaptive-RAG Learning to Adapt Retrieval-Augmented Large Language Models through Question Complexity.pdf>)
- **Authors:** Soyeong Jeong, Jinheon Baek, Sukmin Cho, Sung Ju Hwang, Jong C. Park
- **Venue:** Proceedings of NAACL 2024 (Volume 1: Long Papers), pp. 7036-7050, Mexico City, Mexico, ACL
- **Verdict:** CORE/SUPPORTING (router)
- **RQ:** RQ2 (Router), PROD
- **DOI:** [10.18653/v1/2024.naacl-long.389](https://doi.org/10.18653/v1/2024.naacl-long.389)

**Method.** Trains a smaller classifier to predict query complexity and routes each query to no-retrieval, single-step retrieval, or multi-step retrieval. The classifier is trained on automatically collected labels from the actual outcomes of the candidate strategies.

**Key findings.** Improves the accuracy-efficiency trade-off over single-step, multi-step, and adaptive-retrieval baselines across open-domain QA datasets of varying complexity.

**Use for my work.** Direct justification for my Router query engine and for reporting accuracy/latency per complexity bucket. My synthetic dataset already has single-hop/multi-hop labels, which can supervise or at least evaluate a router.

**Caveats.** English open-domain QA that requires a trained classifier, not regulations, and routing errors propagate.

**Placement in my thesis.** RQ2 Router design and PROD (accuracy vs latency by complexity).

### P14 · `asai2024selfrag`

**Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection**

- **File:** [`papers/pdf/Self-RAG Learning to Retrieve, Generate, and Critique through Self-Reflection.pdf`](<papers/pdf/Self-RAG Learning to Retrieve, Generate, and Critique through Self-Reflection.pdf>)
- **Authors:** Akari Asai, Zeqiu Wu, Yizhong Wang, Avirup Sil, Hannaneh Hajishirzi
- **Venue:** The Twelfth International Conference on Learning Representations (ICLR 2024)
- **URL:** <https://proceedings.iclr.cc/paper_files/paper/2024/hash/25f7be9694d7b32d5cc670927b8091e1-Abstract-Conference.html>
- **Verdict:** SUPPORTING
- **RQ:** RQ2 (adaptive/self-reflective retrieval)

**Method.** Trains a single LM to generate reflection tokens that decide whether to retrieve, whether a passage is relevant, and whether the generation is supported. Retrieval is adaptive and on-demand. The model critiques its own output.

**Key findings.** 7B and 13B Self-RAG models outperform ChatGPT and Llama2-chat on open-domain QA, reasoning, and fact-verification tasks, with significant gains in citation accuracy for long-form generation.

**Use for my work.** Related-work anchor for adaptive retrieval and self-critique. I cite it as a contrast to fixed top-k pipelines. Optional if I later add a self-reflection engine.

**Caveats.** Requires a specially trained critic/reflection-token model, is not plug-and-play with an arbitrary LLM, and adds training and inference cost.

**Placement in my thesis.** RQ2 related work.

## D. Domain and education context

### P15 · `li2025retrieval`

**Retrieval-augmented generation for educational application: A systematic survey**

- **File:** [`papers/pdf/1-s2.0-S2666920X25000578-main.pdf`](<papers/pdf/1-s2.0-S2666920X25000578-main.pdf>)
- **Authors:** Zongxi Li, Zijian Wang, Weiming Wang, Kevin Hung, Haoran Xie, Fu Lee Wang
- **Venue:** Computers and Education: Artificial Intelligence, 8:100417, 2025
- **Verdict:** SUPPORTING
- **RQ:** Introduction/related work
- **DOI:** [10.1016/j.caeai.2025.100417](https://doi.org/10.1016/j.caeai.2025.100417)

**Method.** Systematic survey of RAG in educational applications: architectures, application areas, evaluation practices, and open challenges (hallucination, static knowledge, domain adaptation).

**Key findings.** Consolidates the evidence that RAG is used across teaching, learning, and student-support tasks, and that domain-specific grounding is the main motivation. Identifies evaluation and adaptation as open problems.

**Use for my work.** Framing the education/admissions domain as high-stakes and under-served. A broad citation for "RAG in education" in my introduction. A source for gap statements.

**Caveats.** Survey, not an empirical comparison, not specific to admission regulations, and does not evaluate retrieval architectures.

**Placement in my thesis.** Introduction and related work (education).

### P16 · `virabyan2025enhancing`

**Enhancing Admission Inquiry Responses with Fine-Tuned Models and Retrieval-Augmented Generation**

- **File:** [`papers/pdf/2601.04206v1.pdf`](<papers/pdf/2601.04206v1.pdf>)
- **Authors:** Aram Virabyan
- **Venue:** Proceedings of the 19th International Scientific Conference "Parallel Computing Technologies" (PCT'2025), pp. 99-106, South Ural State University, 2025
- **Verdict:** SUPPORTING
- **RQ:** Domain. Generation adaptation
- **DOI:** [10.14529/pct2025](https://doi.org/10.14529/pct2025)

**Method.** Fine-tuned LLM + RAG for admission inquiries. Compares fine-tuned RAG against RAG-only, fine-tuned-only, and baseline models on response quality and speed.

**Key findings.** Fine-tuned RAG is best overall: 92.7% fact recall and 8.9/10 user satisfaction. RAG-only reaches 91.4% fact recall. Fine-tuned-only drops to 48.3%, showing that fine-tuning cannot supply volatile factual knowledge. Fine-tuning mainly improves contextual understanding and style.

**Use for my work.** Domain motivation and evidence that retrieval, not parametric memory, must carry volatile admission facts. It also justifies my decision to keep the LLM fixed and compare retrieval components (the paper instead adapts the generator).

**Caveats.** Small evaluation at a single institution. Fine-tuning is outside my fixed-control design, and there is no retrieval-component ablation.

**Placement in my thesis.** Related work (domain) and discussion/limitations.

## E. External references

### E01 · `chen2024m3`

**BGE-M3 / M3-Embedding**

- **File:** [`papers/pdf/chen2024m3-embedding.pdf`](<papers/pdf/chen2024m3-embedding.pdf>)
- **Venue:** Findings of ACL 2024, pp. 2318-2335.
- **DOI:** [10.18653/v1/2024.findings-acl.137](https://doi.org/10.18653/v1/2024.findings-acl.137)

**Why it matters.** This is my base embedding model. It supports dense, sparse, and multi-vector retrieval in one model over 100+ languages and up to 8,192 tokens. I cite it in the methods section and when justifying the dense/sparse arms. Caveat: the paper reports multilingual retrieval benchmarks, not Romanian admission regulations.

### E02 · `auer2024docling`

**Docling Technical Report**

- **File:** [`papers/pdf/auer2024docling-technical-report.pdf`](<papers/pdf/auer2024docling-technical-report.pdf>)
- **Venue:** arXiv:2408.09869, 2024.
- **DOI:** [10.48550/arXiv.2408.09869](https://doi.org/10.48550/arXiv.2408.09869)

**Why it matters.** It documents my ingestion layer: layout analysis (DocLayNet) and table structure recognition (TableFormer), with efficient CPU operation. I cite it for the fixed ingestion control and for preserving tables/structure. Caveat: technical report, with the 2025 paper as the updated toolkit description.

### E03 · `livathinos2025docling`

**Docling: An Efficient Open-Source Toolkit**

- **File:** [`papers/pdf/livathinos2025docling-toolkit.pdf`](<papers/pdf/livathinos2025docling-toolkit.pdf>)
- **Venue:** arXiv:2501.17887, 2025 (AAAI-25 Workshop on Open-Source AI for Mainstream Use).
- **DOI:** [10.48550/arXiv.2501.17887](https://doi.org/10.48550/arXiv.2501.17887)

**Why it matters.** The newer, broader Docling paper: unified structured document representation, modular architecture, and integrations (LangChain, LlamaIndex, spaCy). I use it alongside or instead of the technical report when describing the toolkit as a whole.

### E04 · `es2024ragas`

**RAGAs**

- **File:** [`papers/pdf/es2024ragas.pdf`](<papers/pdf/es2024ragas.pdf>)
- **Venue:** EACL 2024 System Demonstrations, pp. 150-158.
- **DOI:** [10.18653/v1/2024.eacl-demo.16](https://doi.org/10.18653/v1/2024.eacl-demo.16)

**Why it matters.** The evaluation framework for Phase 1 (faithfulness, answer relevancy, context precision/recall). I cite it for the metric definitions and for the reference-free evaluation stance. Caveat: LLM-judge metrics are model-sensitive, so I pair them with my deterministic metrics.

### E05 · `upadhyay2024umbrela`

**UMBRELA**

- **File:** [`papers/pdf/upadhyay2024umbrela.pdf`](<papers/pdf/upadhyay2024umbrela.pdf>)
- **Venue:** arXiv:2406.06519, 2024.
- **DOI:** [10.48550/arXiv.2406.06519](https://doi.org/10.48550/arXiv.2406.06519)

**Why it matters.** The open-source LLM relevance assessor used by open-rag-eval in my harness. I cite it for the golden-free judged relevance stage (precision@k, AP@k, MRR). Note the spelling: UMBRELA, not "UMBRELLA".

### E06 · `nogueira2019passage`

**Passage Re-ranking with BERT**

- **File:** [`papers/pdf/nogueira2019passage-reranking-bert.pdf`](<papers/pdf/nogueira2019passage-reranking-bert.pdf>)
- **Venue:** arXiv:1901.04085, 2019.
- **DOI:** [10.48550/arXiv.1901.04085](https://doi.org/10.48550/arXiv.1901.04085)

**Why it matters.** The foundational cross-encoder reranker. I cite it when justifying the cross-encoder arm (Cohere/BGE) and the two-stage retrieve-then-rerank design.

### E07 · `baai2024bgereranker`

**BGE Reranker v2 (bge-reranker-v2-m3)**

- **File:** none (model card)
- **Venue:** Hugging Face model card, 2024.

**Why it matters.** The open reranker implementation used in my BGE reranker arm. I cite it as @misc for the exact model. Caveat: model card, not a peer-reviewed paper.

### E08 · `cohere2025rerank`

**Cohere Rerank**

- **File:** none (documentation)
- **Venue:** Cohere documentation, 2025.

**Why it matters.** The commercial reranker used in my Cohere arm. I cite it as @misc for the API/model. Caveat: vendor documentation. For a peer-reviewed anchor, I cite Nogueira & Cho and the BGE-M3 paper.

## F. Cross-cutting notes for my thesis

### 1. Closest prior systems

- Khasanova & Suh (2025, Electronics) is the closest experimental design: retrieval x chunking x LLM + RAGAS + latency on international admissions.
- Setiawan et al. (2026, Bit-Tech) is the closest reasoning-model study: reasoning vs non-reasoning LLMs under fixed retrieval for admissions.
- Elchafei et al. (2026, SemEval) is the closest parent-child + multi-turn method paper.
- Nguyen & Quan (2025, SOICT) is the closest admission-domain hybrid RAG system.
- Wojtasik et al. (2025, ICCS) is the closest same-domain system: university regulations, bge-m3 + reranker, citations, and an annotated IR/QA dataset. It studies document length, summarization, context size, prompting, and citation accuracy, not my retriever/query-engine matrix or production Pareto.

I should not claim "first comparison of retrieval/chunking for admissions." My claim is the combination: Docling AST hierarchical + parent-child + metadata injection, BGE-M3 dense+sparse, the query-engine suite, multi-turn, and a production Pareto frontier on a Romanian corpus.

### 2. Hypotheses supported by the literature

- **H1 (RQ1):** Structure-aware/hierarchical chunking beats fixed-size and semantic chunking on structured documents (Taiwo, HiChunk, Yang). Summary-only enrichment trades recall for precision (Wasson).
- **H2 (RQ2):** Hybrid retrieval + cross-encoder reranking is the strongest single pipeline (Akarsu, Best Practices, Electronics 2025).
- **H3 (RQ2):** BM25 is a strong baseline for exact figures, and HyDE is more useful for vocabulary mismatch than for numeric lookup (Akarsu, HyDE). I test both query profiles separately.
- **H4 (RQ2/PROD):** Adaptive routing improves the accuracy-latency trade-off (Adaptive-RAG, CRAG), but adds classifier/evaluator cost.
- **H5 (RQ3):** Child-level retrieval + parent-level context reconstruction helps multi-turn faithfulness (H-RAG). I compare query rewriting with retrieval carry-forward.
- **H6 (PROD):** Accuracy gains usually cost latency (Electronics 2025, Taiwo, Best Practices), so I report a Pareto frontier rather than a single "best" configuration.

### 3. Measurement cautions

- Yang's "measurement trap": if I ablate title-chain/heading context, I should not ask annotators to judge relevance without it. Their agreement collapses.
- I match retrieved chunks to golden chunks by chunkID. I keep the chunking fixed so IDs stay stable across retrieval configurations.
- I report per-complexity (single-hop, multi-hop, modified) and per-language results. Aggregate scores hide the router and HyDE effects.
- I keep the ingestion and embedding layers fixed when comparing retrievers. Otherwise the comparison is confounded (this is my thesis design rationale).
