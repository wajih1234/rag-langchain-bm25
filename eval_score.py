import json
import os

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from ragas import EvaluationDataset, evaluate
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithoutReference,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from embd import embeddings

load_dotenv()

# Judge LLM: same OpenRouter setup as llm.py, but temperature 0 and a higher token limit
judge = ChatOpenAI(
    model="meta-llama/llama-3.3-70b-instruct",
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1",
    temperature=0,
    max_tokens=1000,
)

with open("eval_data.json", encoding="utf-8") as f:
    rows = json.load(f)

dataset = EvaluationDataset.from_list(rows)

result = evaluate(
    dataset=dataset,
    metrics=[
        Faithfulness(),
        ResponseRelevancy(),
        LLMContextPrecisionWithoutReference(),
    ],
    llm=LangchainLLMWrapper(judge),
    embeddings=LangchainEmbeddingsWrapper(embeddings),
    run_config=RunConfig(max_workers=1, timeout=180, max_retries=10),
)

df = result.to_pandas()
df.to_csv("eval_results.csv", index=False)
print(df.drop(columns=["retrieved_contexts", "response"], errors="ignore"))
print(result)