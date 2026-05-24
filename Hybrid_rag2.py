import langchain_community.utils.math as math_utils
import numpy as np

_original_cosine_similarity = math_utils.cosine_similarity

def safe_cosine_similarity(X, Y):
    result = _original_cosine_similarity(X, Y)
    return np.array(result, dtype=np.float32)

math_utils.cosine_similarity = safe_cosine_similarity

import json
import gc
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.run_config import RunConfig
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_text_splitters import RecursiveCharacterTextSplitter, TextSplitter
from langchain.retrievers import ParentDocumentRetriever
from langchain.storage import InMemoryStore
from langchain.chains import RetrievalQA

# --- THE FIXES ---

class AbsoluteSemanticChunker(SemanticChunker):
    """Bypasses the percentile trap by strictly splitting on a fixed cosine distance."""
    def __init__(self, embeddings, similarity_threshold=0.85, **kwargs):
        super().__init__(embeddings=embeddings, breakpoint_threshold_type="percentile", **kwargs)
        self.distance_threshold = 1.0 - similarity_threshold

    def _calculate_breakpoint_indices(self, distances):
        breakpoints = []
        for i, distance in enumerate(distances):
            if distance > self.distance_threshold:
                breakpoints.append(i)
        return breakpoints

class SemanticTextSplitterAdapter(TextSplitter):
    """Wraps the chunker to satisfy ParentDocumentRetriever's type checks."""
    def __init__(self, semantic_chunker):
        super().__init__()
        self._chunker = semantic_chunker

    def split_text(self, text: str) -> list[str]:
        docs = self._chunker.create_documents([text])
        return [doc.page_content for doc in docs]

PDF_PATH = "IoT_Fundamentals.pdf"

def build_hybrid_pipeline(pdf_path: str):
    print("Loading PDF...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    # FIX 1: Use from_tiktoken_encoder for true token sizing, keeping the 100 overlap
    parent_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1024,
        chunk_overlap=100 
    )

    # FIX 2: Swap to the Absolute Chunker
    _semantic_chunker = AbsoluteSemanticChunker(
        embeddings=embeddings,
        similarity_threshold=0.85
    )
    semantic_child_splitter = SemanticTextSplitterAdapter(_semantic_chunker)

    print("Initializing vector store and document store...")
    vectorstore = Chroma(
        collection_name="hybrid_rag",
        embedding_function=embeddings
    )
    store = InMemoryStore()

    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=semantic_child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3}
    )

    print("Ingesting documents...")
    for i, doc in enumerate(docs):
        if not doc.page_content.strip():
            continue
            
        # FIX 3: Removed the len < 100 character skip that was dropping pages

        try:
            retriever.add_documents([doc])
        except Exception as e:
            print(f"Skipping page {i+1} due to error: {e}")
            continue

        if (i + 1) % 10 == 0 or (i + 1) == len(docs):
            print(f"Processed {i+1}/{len(docs)} pages...")

    print("Initializing Llama 3.1 8B...")
    llm = Ollama(model="llama3.1:8b")

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

if __name__ == "__main__":
    with open("eval_questions.json", "r") as f:
        qa_pairs = json.load(f)

    user_inputs = [item["question"] for item in qa_pairs]
    references = [item["reference"] for item in qa_pairs]

    ragas_llm = LangchainLLMWrapper(Ollama(model="llama3.1:8b", timeout=600))
    ragas_embeddings = LangchainEmbeddingsWrapper(
        SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    )

    hybrid_chain = build_hybrid_pipeline(PDF_PATH)

    retrieved_contexts = []
    responses = []

    for i, question in enumerate(user_inputs):
        print(f"Running question {i+1}/{len(user_inputs)}...")
        result = hybrid_chain.invoke(question)
        responses.append(result["result"])
        retrieved_contexts.append(
            [doc.page_content for doc in result["source_documents"]]
        )

    del hybrid_chain
    gc.collect()

    dataset = Dataset.from_dict({
        "user_input": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "response": responses,
        "reference": references
    })

    scores = evaluate(
        dataset,
        metrics=[faithfulness, answer_relevancy, context_recall, context_precision],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
        run_config=RunConfig(timeout=600, max_workers=1)
    )

    print("\nHybrid Chunking Results:")
    print(scores)