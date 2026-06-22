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

Nobody had run all four strategies, Fixed-size, Semantic, Hierarchical, and Hybrid, in one controlled pipeline, on a structured technical textbook, using a locally-run open-source model. That's the gap this paper fills.

### Research Questions

1. Do different chunking strategies produce significant performance differences in RAG-based QA on IoT textbooks?
2. Which strategy, Fixed-size, Semantic, Hierarchical, or Hybrid, achieves the highest overall RAGAS score?
3. How does each strategy perform across the four individual RAGAS dimensions?

## Dataset

*IoT Fundamentals: Networking Technologies, Protocols, and Use Cases for the Internet of Things* by David Hanes, Gonzalo Salgueiro, Patrick Grossetete, Robert Barton, and Jerome Henry (Cisco Press, 2017).

Chosen specifically because its chapter/section/subsection structure stress-tests hierarchical chunking in a way generic text can't. The book had never been used before in chunking-comparison research, which also covers the novelty angle. Text was extracted with PyMuPDF, then run through each strategy's processing pipeline independently.

Note: the textbook itself is commercially published and isn't redistributed in the repo. Only the test questions, ground-truth answers, and experiment code are public.

## Experimental Setup

Everything in the pipeline was held constant except the chunking strategy itself, that's the entire point of a controlled comparison. Same embedding model, same generative model, same vector database, same 10-question test set per strategy (40 questions total). The only variable that moved was how text got split.

- **Execution Environemnt:** Google Colab (cloud-hosted VM). All four pipelines ran on the same Colab session to keep hardware conditions consistent across strategies. Ollama was used to serve Llama 3.1 (8B) directly inside the Colab runtime, so there were still no external commercial API calls involved.
Generative model: Llama 3.1 (8B), served via Ollama. No external API calls, no commercial LLM dependency.
- **Generative model:** Llama 3.1 (8B), run locally via Ollama. No external API calls, no commercial LLM dependency.
- **Evaluation framework:** RAGAS, across Faithfulness, Answer Relevance, Context Precision, and Context Recall.
- **Chunking library:** LangChain.

## The Four Strategies

**Fixed-size** (`fixed_rag.py`) — the baseline. 512-token chunks, 50-token overlap, no awareness of sentence or section boundaries. Simple, cheap, and exactly as naive as it sounds.

**Semantic** (`semantic_rag.py`) — boundaries are placed where sentence-embedding similarity drops below 0.85, which is meant to catch topic shifts. Chunk sizes vary, but each one stays thematically coherent. Costs more compute than fixed-size because of the embedding step.

**Hierarchical** (`hierarchical_rag.py`) — two-level structure. Child chunks (256 tokens) handle retrieval precision; parent chunks (1024 tokens) get passed to the model for fuller context once a relevant child chunk is found. Built for documents that already have a layered structure, like, say, a textbook.

**Hybrid** (`hybrid_rag.py`) — combines the other two ideas instead of picking one. Built on LangChain's `ParentDocumentRetriever`, storing both parent and child chunks together. Child chunks are generated with `AbsoluteSemanticChunker`, an extension of LangChain's standard `SemanticChunker` that sets boundaries using percentile limits of cosine distance rather than a flat similarity cutoff. A fixed cosine threshold of 0.15 was used to keep chunk sensitivity consistent across the gradual topic shifts common in IoT textbook content (equivalent to the 0.85 similarity value used in the standalone Semantic strategy). Parent chunks were built with `RecursiveCharacterTextSplitter` at 1024 tokens with 100-token overlap, counted via Tiktoken for accuracy. At retrieval time, the three most relevant child chunks get pulled, and their corresponding parent chunks go to Llama 3.1 as generation context.

The idea behind Hybrid, in short: semantic chunking handles precision at the boundary level, hierarchical structure handles context width at generation time. Neither one alone does both.

## Results

| Chunking Strategy | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---|---|---|---|
| Fixed-size (Baseline) | 0.5717 | 0.5446 | 0.8667 | 0.7800 |
| Semantic | 0.7006 | 0.6396 | 0.9667 | 0.9000 |
| Hierarchical | 0.6958 | 0.7016 | 0.9333 | 0.9050 |
| **Hybrid** | **0.7970** | **0.7848** | **1.0000** | **0.9290** |

Hybrid won every metric. Context Precision came out at a perfect 1.0000, which is the kind of number that should make you suspicious until you check the recall figure too, and the recall figure backs it up at 0.9290, the highest of the four.

### Why Fixed-size lost

512-token cuts with no structural awareness routinely split a thought in half, leaving neither half independently retrievable for a relevant query. Context Recall reflects that gap directly: 0.7800 against 0.9050 for Hierarchical and 0.9000 for Semantic isn't a small difference.

### Why Semantic and Hierarchical each won on different things

Semantic's 0.9667 Context Precision was the strongest of the two non-hybrid approaches. Boundary detection at 0.85 similarity tends to keep chunks on a single topic, which cuts down retrieval noise almost by definition.

Hierarchical's 0.9050 Context Recall came from a more structural decision. Even when the triggering child chunk was narrow, the 1024-token parent chunk it pulled in covered enough surrounding material to actually answer the question.

### Why Hybrid won everything

It gets the noise suppression that Semantic chunking provides at the child-chunk level, which is what pushed Context Precision to a perfect score, and it gets the wide context window from parent chunks at generation time, which is what pushed Context Recall past both standalone strategies. Faithfulness (0.7970) and Answer Relevance (0.7848) followed the same pattern: cleaner, less noisy context arriving at the model produced fewer unsupported claims and answers that actually addressed the question asked.

On Faithfulness specifically, the pattern across all four strategies says something about how Llama 3.1 behaves under incomplete context. When retrieved context is fragmented, the model doesn't flag the gap, it just fills it from parametric memory, and that's where unsupported claims sneak in. Fixed-size's 0.5717 Faithfulness score is less a model failure and more a symptom of what it was being fed.

## Conclusion

Chunking strategy makes a measurable, significant difference in RAG performance on structured technical documents. That answers RQ1 directly. Fixed-size underperformed across every single metric with no close calls. Semantic and Hierarchical each had genuine strengths depending on what you're optimizing for, Semantic for precision-sensitive use cases, Hierarchical for completeness-focused educational QA. Hybrid beat both by combining what each does well, landing the highest score on all four RAGAS dimensions, RQ2 and RQ3 answered together.

For hierarchically organized educational content specifically, retrieval quality looks like the primary driver of end-to-end answer quality. A strategy that handles boundary precision and context width at the same time outperforms anything that only handles one.

## Future Work

- Re-run all four strategies against commercial LLM APIs (Gemini, Claude) to see whether the performance gaps hold, or whether a stronger generative model can compensate for weaker chunking.
- Swap the general-purpose embedding model for something IoT-domain-specific.
- Run a parameter sensitivity sweep on the Hybrid configuration, varying the semantic similarity threshold and parent chunk size, to check how stable these results are outside the exact hyperparameters used here.

## Repository Structure

```
.
├── fixed_rag.py          # Fixed-size chunking pipeline
├── semantic_rag.py        # Semantic chunking pipeline
├── hierarchical_rag.py    # Hierarchical (parent-child) chunking pipeline
├── hybrid_rag.py          # Hybrid chunking pipeline (Semantic + Hierarchical)
├── questions/              # 40 test questions with ground-truth reference answers
└── results/                 # RAGAS evaluation outputs per strategy
```

## Data Availability

Test questions, ground-truth answers, and experiment code: [github.com/Davinchii53/Research_LLM_Methodology](https://github.com/Davinchii53/Research_LLM_Methodology.git)

The source textbook is commercially published by Cisco Press and is not redistributed here.

## Acknowledgments

Thanks to Pak Johan Muliadi Kerta for review and feedback throughout this research. AI-assisted tools used during the research process include Claude as a coding assistant and Google Gemini for research finding support.

## Citation

If you use this work, please cite the paper: *Comparative Study of Text Chunking Strategies for RAG-Based Questions and Answer Systems in IoT Textbooks*, Anshary, K. N., Ramadhan, D. P. F., Saputra, S. H. O., & Kerta, J. M.
