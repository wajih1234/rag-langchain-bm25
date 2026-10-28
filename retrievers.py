from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever


def build_hybrid_retriever(chunks, vectorstore, k=4):
    bm25 = BM25Retriever.from_documents(chunks)
    bm25.k = k

    vector = vectorstore.as_retriever(search_kwargs={"k": k})

    return EnsembleRetriever(
        retrievers=[bm25, vector],
        weights=[0.5, 0.5],
    )


if __name__ == "__main__":
    from ingest import load_file, split_docs
    from embd import build_vectorstore

    chunks = split_docs(load_file("data/llm.pdf"))
    vs = build_vectorstore(chunks)
    retriever = build_hybrid_retriever(chunks, vs)

    results = retriever.invoke("What is deductive coding?")
    for r in results:
        print(r.metadata.get("page"), "|", r.page_content[:150].replace("\n", " "))