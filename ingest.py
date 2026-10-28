from pathlib import Path
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_file(path):
    ext = Path(path).suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(path).load()
    if ext == ".txt":
        return TextLoader(path, encoding="utf-8").load()
    raise ValueError(f"Type de fichier non supporté : {ext}")
def split_docs(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    return splitter.split_documents(docs)

if __name__ == "__main__":
    docs = load_file("data/llm.pdf")
    chunks=split_docs(docs)
    print(len(chunks))
    print(chunks[0].page_content[:200])