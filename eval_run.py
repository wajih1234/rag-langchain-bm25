import json

from ingest import load_file, split_docs
from embd import build_vectorstore
from retrievers import build_hybrid_retriever
from rag import ask

chunks = split_docs(load_file("data/llm.pdf"))
vs = build_vectorstore(chunks)
retriever = build_hybrid_retriever(chunks, vs)

with open("eval_questions.json", encoding="utf-8") as f:
    questions = json.load(f)

rows = []
for q in questions:
    answer, docs = ask(retriever, q)
    rows.append(
        {
            "user_input": q,
            "response": answer,
            "retrieved_contexts": [d.page_content for d in docs],
        }
    )
    print(f"done: {q}")

with open("eval_data.json", "w", encoding="utf-8") as f:
    json.dump(rows, f, ensure_ascii=False, indent=2)