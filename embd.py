import uuid
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def build_vectorstore(chunks):
    return Chroma.from_documents(
        chunks, embeddings, collection_name=str(uuid.uuid4())
    )


if __name__ == "__main__":
    from ingest import load_file, split_docs

    chunks = split_docs(load_file("data/llm.pdf"))
    vs = build_vectorstore(chunks)
    print(vs._collection.count())