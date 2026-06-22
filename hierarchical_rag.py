from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import InMemoryStore

def build_hierarchical_pipeline(pdf_path):
    print("Loading PDF for Hierarchical Chunking...")
    loader = PyMuPDFLoader(pdf_path)
    docs = loader.load()
    print(f"Loaded {len(docs)} pages")

    # Define the splitters (chunk generation is handled by the retriever now)
    parent_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=0)
    child_splitter = RecursiveCharacterTextSplitter(chunk_size=256, chunk_overlap=0)

    print("Initializing embedding model...")
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

    print("Initializing ephemeral vector store and in-memory document store...")
    # Chroma handles the child chunks
    vectorstore = Chroma(
        collection_name="hierarchical_rag",
        embedding_function=embeddings
    )
    
    # InMemoryStore handles the parent chunks
    store = InMemoryStore()

    # Initialize the true hierarchical retriever
    retriever = ParentDocumentRetriever(
        vectorstore=vectorstore,
        docstore=store,
        child_splitter=child_splitter,
        parent_splitter=parent_splitter,
        search_kwargs={"k": 3}
    )

    print("Ingesting documents via micro-batches to prevent ChromaDB crash...")
    # Feeding pages one by one limits the per-transaction child chunk count
    # safely below the 5461 batch limit.
    for i, doc in enumerate(docs):
        # THE FIX: Skip empty pages to prevent the ChromaDB empty embeddings error
        if not doc.page_content.strip():
            continue
            
        retriever.add_documents([doc])
        if (i + 1) % 10 == 0 or (i + 1) == len(docs):
            print(f"Processed {i + 1}/{len(docs)} pages...")

    print("Initializing Llama 3.1 8B and RetrievalQA chain...")
    llm = Ollama(model="llama3.1:8b")
    
    # The retriever now outputs the 1024-token parent chunks to the LLM
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )

    return qa_chain

if __name__ == "__main__":
    chain = build_hierarchical_pipeline("IoT_Fundamentals.pdf")
    response = chain.invoke("What are the key components of an IoT network?")
    print("\nResponse:\n", response['result'])