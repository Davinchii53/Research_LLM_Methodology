---
inclusion: always
---

# Project context: RAG chunking paper

This repo backs an accepted IEEE conference paper: *Comparative Study of Text Chunking Strategies for RAG-Based Question Answering on an IoT Textbook* (Anshary, Ramadhan, Saputra, Kerta; Bina Nusantara University). It is to be presented at IITSOE 2026 on 29 October 2026. The camera-ready paper is final, so treat its numbers and claims as fixed.

## Ground truth: values in the published paper

Table III, means over 10 questions with one run per configuration:

| Configuration | Faithfulness | Answer Relevance | Context Precision | Context Recall |
|---|---|---|---|---|
| Fixed-size | 0.5717 | 0.5446 | 0.8667 | 0.7800 |
| Semantic | 0.7006 | 0.6396 | 0.9667 | 0.9000 |
| Hierarchical | 0.6958 | 0.7016 | 0.9333 | 0.9050 |
| Hybrid | 0.7970 | 0.7848 | 1.0000 | 0.9290 |

Facts already verified against the code. Don't re-derive them, and don't "correct" them:

- **Chunk sizes are characters, not tokens**, for Fixed-size (512, overlap 50) and Hierarchical (child 256, parent 1024). Both use plain `RecursiveCharacterTextSplitter`, which counts with `len`. Only the Hybrid parent (`Hybrid_rag2.py`) uses `.from_tiktoken_encoder` and is 1024 tokens with 100-token overlap.
- **Semantic and Hybrid share the same chunker.** Both use `AbsoluteSemanticChunker` with `similarity_threshold=0.85` (cosine distance > 0.15), which replaces `SemanticChunker`'s percentile breakpoints with a fixed threshold.
- `rerun_semantic.py` is an evaluation driver. The pipeline it runs is in `semantic_rag.py`.
- The RAGAS judge is Llama 3.1 8B via Ollama, the same model that generated the answers. Temperature is left at the default and no seed is set.
- All reported runs were executed on Google Colab with Ollama running inside the runtime.
- The textbook PDF (`IoT_Fundamentals.pdf`) is copyrighted. It is gitignored and must never be committed.

## Open tasks (these need files on the author's machine)

The cloud agent couldn't do these because the run outputs exist only locally.

1. **Find the original run outputs.** Look for the Colab notebook (`.ipynb`), exported logs, or saved console output from the four RAGAS runs.
2. **Check the scores against Table III.** For each configuration, compare the printed RAGAS means with the table above. Report every mismatch exactly. Never adjust the numbers to make them match.
3. **Check the Hybrid run for skipped pages.** `Hybrid_rag2.py` (lines 103–105) catches errors while ingesting a page, prints `Skipping page N due to error: ...`, and continues. Search the Hybrid output for that string and report the page numbers, or confirm none appeared. If any pages were skipped, Hybrid ran on less of the book than the other three, and the author needs to know.
4. **Save the outputs to the repo** under `results/`, one file per configuration, plus the notebook if it exists. Keep them exactly as produced. Don't regenerate or reformat them. Strip anything that could identify the textbook's copyrighted text beyond short retrieved snippets, and never include the PDF.
5. **Update the README** to point to `results/` and to state plainly what the files are. If the outputs can't be found, add one sentence instead: per-question outputs were not saved, and Table III reports the printed means.

## Small pending fixes (low priority)

- `hierarchical_rag.py` line 57: the comment says "1024-token parent chunks", but the splitter counts characters. Fix the comment only; don't change the code.
- README Authors section: Pak Johan is listed as "Supervisor", while the paper lists him as fourth author. Use "Fourth author and supervisor".

## Rules

- **Do not re-run the experiments** unless the author explicitly asks. A re-run with unseeded generation will produce different numbers, which don't replace the published ones.
- **Do not change experimental code behaviour.** Comment and documentation fixes are fine.
- Don't make claims the evidence doesn't support. If something can't be verified, say so.
- Work on a branch and open a pull request into `main`. Never push directly to `main`.
