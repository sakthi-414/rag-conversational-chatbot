import os
import streamlit as st
from ingest import ingest
from agent import app,initial_state

st.set_page_config(page_title="RAG Chatbot", page_icon="🤖")
st.title("RAG Chatbot")

os.makedirs("data", exist_ok=True)

st.sidebar.header("Upload PDF")

if "kb_ready" not in st.session_state:
    st.session_state.kb_ready = False

def sidebar():
    st.sidebar.title("Knowledge Base")

    uploaded_file = st.sidebar.file_uploader(
        "Upload PDF",
        type=["pdf"]
    )

    ingest_clicked = st.sidebar.button(
        "📥 Ingest",
        disabled=(uploaded_file is None)
    )

    return uploaded_file, ingest_clicked

uploaded_file, ingest_clicked = sidebar()

if ingest_clicked:
    save_path=os.path.join("data",uploaded_file.name)
    try:
        with open(save_path,"wb") as f:
            f.write(uploaded_file.getbuffer())
        with st.spinner("Building your knowledge base...."):
            chunks_count=ingest(save_path)
        st.session_state.kb_ready = True
        st.sidebar.success(
    f"Knowledge Base created successfully!\n\n"
    f"📄 File: {uploaded_file.name}\n"
    f"🧩 Chunks created: {chunks_count}")
    except Exception as e:
        st.sidebar.error(e)
        st.stop()

if "state" not in st.session_state:
    st.session_state.state = dict(initial_state)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

question=st.chat_input("Ask your question....",disabled=(not st.session_state.kb_ready))

if question:
    st.session_state.state["question"]=question
    st.session_state.messages.append({
                "role": "user",
                "content": question
            })
    with st.chat_message("user"):
                st.markdown(question)
    try:
        with st.spinner("Generating answer..."):
            response=app.invoke(st.session_state.state)
            st.session_state.state=response
            answer=response["answer"]
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer})
            with st.chat_message("assistant"):
                st.markdown(answer)
    except Exception as e:
        st.error(e)
        st.stop()
if st.sidebar.button("Clear History",disabled=(len(st.session_state.messages)==0)):
    st.session_state.messages=[]
    st.session_state.state=dict(initial_state)
    st.rerun()

