from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm import llm

prompt = ChatPromptTemplate.from_template(
    """Answer the question using only the context below.
If the answer is not in the context, say you don't know.

Context:
{context}

Question: {question}"""
)


def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)


def build_chain(retriever):
    return (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )


if __name__ == "__main__":
    from ingest import load_file, split_docs
    from embd import build_vectorstore
    from retrievers import build_hybrid_retriever

    chunks = split_docs(load_file("data/llm.pdf"))
    vs = build_vectorstore(chunks)
    retriever = build_hybrid_retriever(chunks, vs)
    chain = build_chain(retriever)

    print(chain.invoke("What is deductive coding?"))