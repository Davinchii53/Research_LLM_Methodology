from langchain_community.document_loaders import PyMuPDFLoader
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_experimental.text_splitter import SemanticChunker

class AbsoluteSemanticChunker(SemanticChunker):
    def __init__(self, embeddings, similarity_threshold=0.85, **kwargs):
        # Pass a standard type to satisfy the parent class's internal validation
        super().__init__(embeddings=embeddings, breakpoint_threshold_type="percentile", **kwargs)
        # Convert the paper's similarity threshold (0.85) to LangChain's distance metric (0.15)
        self.distance_threshold = 1.0 - similarity_threshold

    def _calculate_breakpoint_indices(self, distances):
        """
        Override the default statistical breakpoints.
        Split whenever the cosine distance between sentences exceeds the threshold.
        """
        breakpoints = []
        for i, distance in enumerate(distances):
            if distance > self.distance_threshold:
                breakpoints.append(i)
        return breakpoints

def build_semantic_pipeline(pdf_path):
    print("Loading PDF for Semantic Chunking...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()

    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Applying Absolute Semantic Chunking...")
    # Use the custom absolute chunker to match the paper's methodology
    semantic_splitter = AbsoluteSemanticChunker(
        embeddings=embeddings,
        similarity_threshold=0.85
    )
    semantic_chunks = semantic_splitter.split_documents(docs)
    print(f"Created {len(semantic_chunks)} semantic chunks.")

    vectorstore = Chroma.from_documents(semantic_chunks, embeddings)  # no persist_directory
    print("Vector store created")

    llm = Ollama(model="llama3.1:8b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain

if __name__ == "__main__":
    chain = build_semantic_pipeline("IoT_Fundamentals.pdf")
    response = chain.invoke("What is the definition of IoT?")
    print(f"\nQ: What is the definition of IoT?")
    print(f"A: {response['result']}")