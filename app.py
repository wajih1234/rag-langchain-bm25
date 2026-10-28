import os
import tempfile
import uuid
from pathlib import Path

import streamlit as st
from ingest import load_file, split_docs
from embd import build_vectorstore
from retrievers import build_hybrid_retriever
from rag import build_chain

st.set_page_config(page_title="RAG Chat", layout="wide")


def new_conv():
    return {"title": "New chat", "messages": [], "chain": None, "key": None, "n_chunks": 0}


if "convs" not in st.session_state:
    cid = str(uuid.uuid4())
    st.session_state.convs = {cid: new_conv()}
    st.session_state.cur = cid

convs = st.session_state.convs
cur = st.session_state.cur
conv = convs[cur]

with st.sidebar:
    if st.button("➕ New chat", use_container_width=True):
        cid = str(uuid.uuid4())
        convs[cid] = new_conv()
        st.session_state.cur = cid
        st.rerun()

    st.header("Your files")
    files = st.file_uploader(
        "Drop your files here",
        type=["pdf", "txt"],
        accept_multiple_files=True,
        key=f"up_{cur}",
    )

    if files:
        key = [(f.name, f.size) for f in files]
        if conv["key"] != key:
            docs = []
            for f in files:
                with tempfile.NamedTemporaryFile(delete=False, suffix=Path(f.name).suffix) as tmp:
                    tmp.write(f.getvalue())
                docs.extend(load_file(tmp.name))
                os.remove(tmp.name)

            chunks = split_docs(docs)
            vs = build_vectorstore(chunks)
            conv["chain"] = build_chain(build_hybrid_retriever(chunks, vs))
            conv["n_chunks"] = len(chunks)
            conv["key"] = key
            conv["messages"] = []

    if conv["chain"]:
        st.success(f"{conv['n_chunks']} chunks ready")

    st.header("Conversations")
    for cid, c in convs.items():
        label = ("▶ " if cid == cur else "") + c["title"]
        if st.button(label, key=f"btn_{cid}", use_container_width=True):
            st.session_state.cur = cid
            st.rerun()

st.title("RAG Chat")

if conv["chain"] is None:
    st.info("Upload a file in the sidebar to start chatting.")
    st.stop()

for m in conv["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

if question := st.chat_input("Ask a question about your files"):
    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer = conv["chain"].invoke(question)
        st.write(answer)
    if not conv["messages"]:
        conv["title"] = question[:30]
    conv["messages"].append({"role": "user", "content": question})
    conv["messages"].append({"role": "assistant", "content": answer})
    st.rerun()