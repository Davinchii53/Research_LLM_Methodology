from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain_classic.chains import RetrievalQA

def build_fixed_pipeline(pdf_path):
    print("Loading PDF for Fixed Chunking...")
    loader = PyMuPDFLoader(pdf_path)
    documents = loader.load()
    print(f"Loaded {len(documents)} pages")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=512,
        chunk_overlap=50
    )
    chunks = splitter.split_documents(documents)
    print(f"Total chunks: {len(chunks)}")

    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(chunks, embeddings)
    print("Vector store created")

    llm = Ollama(model="llama3.1:8b")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )

    return qa_chain

if __name__ == "__main__":
    chain = build_fixed_pipeline("IoT_Fundamentals.pdf")
    response = chain.invoke("What is the Internet of Things?")
    print(f"\nQ: What is the Internet of Things?")
    print(f"A: {response['result']}")
