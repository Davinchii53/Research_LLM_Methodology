import json
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall, context_precision

from fixed_rag import build_fixed_pipeline
from hierarchical_rag import build_hierarchical_pipeline
from semantic_rag import build_semantic_pipeline
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.llms import Ollama
from langchain_community.embeddings import SentenceTransformerEmbeddings
from ragas.run_config import RunConfig


#RAGAS To use ollama
ragas_llm = LangchainLLMWrapper(Ollama(
    model="llama3.1:8b",
    timeout=300))
ragas_embeddings = LangchainEmbeddingsWrapper(SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2"))

PDF_PATH = "IoT_Fundamentals.pdf"

# Load questions
with open("eval_questions.json", "r") as f:
    qa_pairs = json.load(f)

user_inputs = [item["question"] for item in qa_pairs]
references = [item["reference"] for item in qa_pairs]

def run_pipeline(pipeline_name, qa_chain):
    print(f"\n{'='*40}")
    print(f"Running: {pipeline_name}")
    print(f"{'='*40}")

    retrieved_contexts = []
    responses = []

    for question in user_inputs:
        result = qa_chain.invoke(question)
        responses.append(result["result"])
        retrieved_contexts.append(
            [doc.page_content for doc in result["source_documents"]]
        )

    dataset = Dataset.from_dict({
        "user_input": user_inputs,
        "retrieved_contexts": retrieved_contexts,
        "response": responses,
        "reference": references
    })

    scores = evaluate(
    dataset,
    metrics=[
        faithfulness,
        answer_relevancy,
        context_recall,
        context_precision
    ],
    llm=ragas_llm,
    embeddings=ragas_embeddings,
    run_config=RunConfig(
        timeout=300,
        max_workers=1
    )
)

    print(f"\nResults for {pipeline_name}:")
    print(scores)
    return scores

# Run all 3 sequentially
fixed_chain = build_fixed_pipeline(PDF_PATH)
fixed_scores = run_pipeline("Fixed Chunking", fixed_chain)

hierarchical_chain = build_hierarchical_pipeline(PDF_PATH)
hierarchical_scores = run_pipeline("Hierarchical Chunking", hierarchical_chain)

semantic_chain = build_semantic_pipeline(PDF_PATH)
semantic_scores = run_pipeline("Semantic Chunking", semantic_chain)

# Summary comparison
print("\n\n=== FINAL COMPARISON ===")
print(f"Fixed:        {fixed_scores}")
print(f"Hierarchical: {hierarchical_scores}")
print(f"Semantic:     {semantic_scores}")