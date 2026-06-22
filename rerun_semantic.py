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

from semantic_rag import build_semantic_pipeline

PDF_PATH = "IoT_Fundamentals.pdf"

# Load questions
with open("eval_questions.json", "r") as f:
    qa_pairs = json.load(f)

user_inputs = [item["question"] for item in qa_pairs]
references = [item["reference"] for item in qa_pairs]

# RAGAS judge config — increase timeout
ragas_llm = LangchainLLMWrapper(Ollama(model="llama3.1:8b", timeout=600))
ragas_embeddings = LangchainEmbeddingsWrapper(
    SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
)

# Build pipeline
semantic_chain = build_semantic_pipeline(PDF_PATH)

# Run questions
retrieved_contexts = []
responses = []

for question in user_inputs:
    result = semantic_chain.invoke(question)
    responses.append(result["result"])
    retrieved_contexts.append(
        [doc.page_content for doc in result["source_documents"]]
    )

del semantic_chain
gc.collect()

# Evaluate
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

print("\nSemantic Chunking (Rerun):")
print(scores)