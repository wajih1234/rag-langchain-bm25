from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from llm import llm
import time
from monitoring import log_query
from retrievers import build_hybrid_retriever


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

def ask(retriever, question):
    start = time.time()
    try:
        docs = retriever.invoke(question)
        answer = (prompt | llm | StrOutputParser()).invoke(
            {"context": format_docs(docs), "question": question}
        )
        log_query(
            question, answer, [d.page_content for d in docs], time.time() - start
        )
        return answer, docs
    except Exception as e:
        log_query(question, None, [], time.time() - start, error=str(e))
        raise
if __name__ == "__main__":
    from ingest import load_file, split_docs
    from embd import build_vectorstore
    from retrievers import build_hybrid_retriever

    chunks = split_docs(load_file("data/llm.pdf"))
    vs = build_vectorstore(chunks)
    retriever = build_hybrid_retriever(chunks, vs)

    answer, docs = ask(retriever, "What is deductive coding?")
    print(answer)