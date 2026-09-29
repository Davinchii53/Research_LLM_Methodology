# Comparative Study of Text Chunking Strategies for RAG-Based Question Answering in IoT Textbooks

A controlled experiment comparing four text chunking strategies, Fixed-size, Semantic, Hierarchical, and Hybrid, inside a single RAG pipeline built on Llama 3.1 (8B). Evaluated with the RAGAS framework across Faithfulness, Answer Relevance, Context Precision, and Context Recall.

This work was submitted as a Research Methodology paper at BINUS University, Bandung Campus.

## Authors

- **Kelvin Nabil Anshary** — Computer Science, BINUS University Bandung. Designed and implemented the full RAG pipeline, all four chunking strategies, and the experimental setup.
- **Deas Putra Fajar Ramadhan** — Computer Science, BINUS University Bandung. Paper writing, literature search, reference verification.
- **Samuel Handyanto Ongko Saputra** — Computer Science, BINUS University Bandung. Paper writing, literature search, reference verification.
- **Pak Johan Muliadi Kerta** — Supervisor, Computer Science, BINUS University Bandung. Research review and feedback throughout the study.

## Background

LLMs hallucinate. They're stuck with whatever knowledge they had at training time, and asking them anything outside that boundary tends to produce confident, wrong answers. Retrieval-Augmented Generation, introduced by Lewis et al. in 2020, became the standard fix: pull relevant chunks from an external knowledge source, hand them to the model, and ground the answer in something real instead of parametric memory alone.

The part of that pipeline nobody pays attention to is chunking. It's the step where a source document gets cut into smaller units before indexing, and it's arguably the least explored piece of the whole RAG workflow despite being the one that decides what information the model even has access to. Get it wrong and a chunk boundary slices straight through the middle of an explanation, the retriever pulls back something incomplete, and the model fills the gap with a guess. That's not a generation problem. That's a chunking problem wearing a generation problem's clothes.

This matters more for structured technical documents than it does for general text. A textbook organizes knowledge into chapters, sections, and subsections on purpose. A naive fixed-size cut doesn't know or care about any of that structure.

## The Problem

Most existing chunking comparisons fall into one of a few traps: they measure retrieval quality only, without checking whether better retrieval actually produces better answers. Or they rely on a commercial LLM, which makes the study hard to reproduce. Or they test on general, loosely structured text instead of something like a textbook with real hierarchical organization.

These strategies have been compared before, and recent work is mixed on whether the sophisticated ones reliably pay off. This study doesn't propose a new chunking method. What it contributes is a controlled comparison in which chunking is the only stage that varies, run end-to-end on answer quality rather than retrieval metrics alone, on a structured technical textbook, with a self-hosted open-weight model and no commercial API — plus the code and question set to rerun it.

### Research Questions

1. Do different chunking strategies produce measurable performance differences in RAG-based QA on an IoT textbook?
2. Which strategy, Fixed-size, Semantic, Hierarchical, or Hybrid, achieves the highest overall RAGAS score?
3. How does each strategy perform across the four individual RAGAS dimensions?

## Dataset

*IoT Fundamentals: Networking Technologies, Protocols, and Use Cases for the Internet of Things* by David Hanes, Gonzalo Salgueiro, Patrick Grossetete, Robert Barton, and Jerome Henry (Cisco Press, 2017).

Chosen specifically because its chapter/section/subsection structure stress-tests hierarchical chunking in a way generic text can't, and because its breadth supports questions across a range of topics. Text was extracted with PyMuPDF, then run through each strategy's processing pipeline independently.

Note: the textbook itself is commercially published and isn't redistributed in the repo. Only the test questions, ground-truth answers, and experiment code are public.

## Experimental Setup

Everything in the pipeline was held constant except the chunking strategy itself, that's the entire point of a controlled comparison. Same embedding model, same generative model, same vector database, and one shared set of 10 questions submitted to all four strategies, giving 40 question-strategy evaluations. The only variable that moved was how text got split.

- **Execution environment:** Google Colab (cloud-hosted VM). The pipeline was developed on local hardware, but every reported run was executed in Colab so hardware conditions are common across all four strategies. Ollama served Llama 3.1 (8B) directly inside the Colab runtime, so this is a self-hosted deployment with no commercial model API at any stage.
- **Generative model:** Llama 3.1 8B (`llama3.1:8b`), served via Ollama. Decoding left at Ollama defaults; temperature not overridden and no random seed set, so generation was stochastic.
- **Embedding model:** `all-MiniLM-L6-v2` (sentence-transformers), used for retrieval in all four strategies and by the RAGAS evaluator.
- **Vector store:** Chroma, ephemeral in-memory collection rebuilt per strategy. Retrieval depth top-k = 3 in all four strategies.
- **Evaluation framework:** RAGAS, across Faithfulness, Answer Relevance, Context Precision, and Context Recall. The judge model is Llama 3.1 8B via Ollama, the same model that generated the answers.
- **Chunking library:** LangChain.

## The Four Strategies

A note on units before the details: `RecursiveCharacterTextSplitter` measures `chunk_size` with Python's builtin `len`, so it counts **characters** unless you construct it with `.from_tiktoken_encoder(...)`. Fixed-size and Hierarchical use the plain constructor and are therefore specified in characters. Only the Hybrid parent splitter counts tokens. The two units are not interchangeable — 512 characters of technical prose is roughly 90-130 tokens.

**Fixed-size** (`fixed_rag.py`) — the baseline. 512-character chunks, 50-character overlap, no awareness of sentence or section boundaries. Simple, cheap, and exactly as naive as it sounds.

**Semantic** (`semantic_rag.py`) — single-level. Boundaries are placed by `AbsoluteSemanticChunker` wherever the cosine distance between adjacent sentence embeddings exceeds 0.15, equivalent to a similarity of 0.85, which is meant to catch topic shifts. Chunk sizes vary, but each one stays thematically coherent. Costs more compute than fixed-size because of the embedding step. Retrieved chunks go straight to the model.

**Hierarchical** (`hierarchical_rag.py`) — two-level structure. Child chunks (256 characters) handle retrieval precision; parent chunks (1024 characters) get passed to the model for fuller context once a relevant child chunk is found. Built for documents that already have a layered structure, like, say, a textbook.

**Hybrid** (`Hybrid_rag2.py`) — combines the other two ideas instead of picking one. Built on LangChain's `ParentDocumentRetriever`, storing both parent and child chunks together. Child chunks are generated with `AbsoluteSemanticChunker`, an extension of LangChain's standard `SemanticChunker`. The standard version sets boundaries at a *percentile* of the cosine-distance distribution within a document, which makes sensitivity depend on the document being processed; the subclass overrides `_calculate_breakpoint_indices` to use a *fixed absolute* cosine-distance threshold of 0.15 instead, keeping chunk sensitivity consistent across the gradual topic shifts common in IoT textbook content. That threshold is the same one the standalone Semantic strategy uses, so the two strategies derive their retrieval units the same way and differ mainly in what reaches the model. Parent chunks were built with `RecursiveCharacterTextSplitter.from_tiktoken_encoder` at 1024 tokens with 100-token overlap. At retrieval time, the three most relevant child chunks get pulled, and their corresponding parent chunks go to Llama 3.1 as generation context.

The idea behind Hybrid, in short: semantic chunking handles precision at the boundary level, hierarchical structure handles context width at generation time. Neither one alone does both.

## Results

| Chunking Strategy | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---|---|---|---|
| Fixed-size (Baseline) | 0.5717 | 0.5446 | 0.8667 | 0.7800 |
| Semantic | 0.7006 | 0.6396 | 0.9667 | 0.9000 |
| Hierarchical | 0.6958 | 0.7016 | 0.9333 | 0.9050 |
| **Hybrid** | **0.7970** | **0.7848** | **1.0000** | **0.9290** |

Hybrid recorded the highest mean on every metric. Context Precision came out at a perfect 1.0000, which is the kind of number that should make you suspicious — on 10 questions it means no irrelevant passage was observed, not that none is possible. Read it alongside the recall figure, which backs the result up at 0.9290, the highest of the four.

These are means over 10 questions with one run per strategy. No significance test was applied, so treat every gap below as descriptive rather than established.

### Why Fixed-size lost

512-character cuts with no structural awareness routinely split a thought in half, leaving neither half independently retrievable for a relevant query. At roughly 90-130 tokens a chunk, there often isn't room for a complete protocol explanation in the first place. Context Recall reflects that gap directly: 0.7800 against 0.9050 for Hierarchical and 0.9000 for Semantic isn't a small difference.

### Why Semantic and Hierarchical each won on different things

Semantic's 0.9667 Context Precision was the strongest of the two non-hybrid approaches. Boundary detection at 0.85 similarity tends to keep chunks on a single topic, which cuts down retrieval noise almost by definition.

Hierarchical's 0.9050 Context Recall came from a more structural decision. Even when the triggering child chunk was narrow, the 1024-character parent chunk it pulled in covered enough surrounding material to actually answer the question.

### Why Hybrid won everything

It gets the noise suppression that Semantic chunking provides at the child-chunk level, which is what pushed Context Precision to a perfect score, and it gets the wide context window from parent chunks at generation time, which is what pushed Context Recall past both standalone strategies. Faithfulness (0.7970) and Answer Relevance (0.7848) followed the same pattern: cleaner, less noisy context arriving at the model produced fewer unsupported claims and answers that actually addressed the question asked.

On Faithfulness specifically, the pattern across all four strategies says something about how Llama 3.1 behaves under incomplete context. When retrieved context is fragmented, the model doesn't flag the gap, it just fills it from parametric memory, and that's where unsupported claims sneak in. Fixed-size's 0.5717 Faithfulness score is less a model failure and more a symptom of what it was being fed.

## Conclusion

Chunking strategy made a measurable difference in RAG performance on this textbook. That answers RQ1, though descriptively rather than statistically. Fixed-size underperformed across every single metric with no close calls. Semantic and Hierarchical each had genuine strengths depending on what you're optimizing for, Semantic for precision-sensitive use cases, Hierarchical for completeness-focused educational QA — their 0.0048 Faithfulness gap is too small to order them. Hybrid recorded the highest score on all four RAGAS dimensions, RQ2 and RQ3 answered together.

For this textbook, retrieval quality looks like the primary influence on end-to-end answer quality. One caveat worth stating plainly: Hybrid also supplied considerably more context to the model than the other strategies, since its parent chunks are sized in tokens rather than characters, so its margin can't be attributed to boundary placement alone. Separating those two factors needs a context-budget-matched control, which this experiment doesn't have.

## Limitations

Worth knowing before you read anything above as a general result:

- **10 questions, one run per strategy.** Small enough that a single hard question moves a mean noticeably, and no significance test was applied.
- **Generation was stochastic.** Temperature was left at the Ollama default and no seed was set, so run-to-run variance is unmeasured.
- **The RAGAS judge is the model under test.** Llama 3.1 8B scored its own output, so the evaluation isn't independent. No human scoring was done.
- **Context budgets weren't equal.** Fixed-size and Hierarchical size chunks in characters; Hybrid's parents are sized in tokens. Hybrid therefore received substantially more context, so its margin isn't attributable to boundary placement alone.
- **One textbook, one domain, one model, fixed hyperparameters.** Nothing here establishes that the ordering generalizes.

## Future Work

- Expand the question set and report per-question scores, so paired significance testing with effect sizes becomes possible.
- Repeat each strategy across multiple runs with temperature fixed and seeds recorded, and report variance alongside means.
- Add human scoring on a subset, and use a judge model that is separate from and stronger than the generator.
- Add a context-budget-matched control so boundary placement can be separated from parent-chunk width.
- Re-run all four strategies against commercial LLM APIs (Gemini, Claude) to see whether the performance gaps hold, or whether a stronger generative model can compensate for weaker chunking.
- Swap the general-purpose embedding model for something IoT-domain-specific.
- Run a parameter sensitivity sweep on the Hybrid configuration, varying the semantic similarity threshold and parent chunk size, to check how stable these results are outside the exact hyperparameters used here.

## Repository Structure

```
.
├── main.py                # Builds and evaluates the Fixed-size, Hierarchical, and Semantic pipelines
├── fixed_rag.py           # Fixed-size chunking pipeline
├── semantic_rag.py        # Semantic chunking pipeline
├── hierarchical_rag.py    # Hierarchical (parent-child) chunking pipeline
├── Hybrid_rag2.py         # Hybrid chunking pipeline, with its own evaluation driver
├── rerun_semantic.py      # Standalone evaluation driver for the Semantic pipeline
├── eval_questions.json    # 10 test questions with ground-truth reference answers
└── .gitignore
```

## Data Availability

Test questions, ground-truth answers, and experiment code: [github.com/Davinchii53/Research_LLM_Methodology](https://github.com/Davinchii53/Research_LLM_Methodology.git)

The source textbook is commercially published by Cisco Press and is not redistributed here.

## Acknowledgments

Thanks to Pak Johan Muliadi Kerta for review and feedback throughout this research. AI-assisted tools used during the research process include Claude as a coding assistant and Google Gemini for research finding support.

## Citation

If you use this work, please cite the paper: *Comparative Study of Text Chunking Strategies for RAG-Based Question Answering on an IoT Textbook*, Anshary, K. N., Ramadhan, D. P. F., Saputra, S. H. O., & Kerta, J. M.
