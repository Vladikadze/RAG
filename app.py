import streamlit as st
from query import ask

st.set_page_config(page_title="Local RAG", page_icon="🔍", layout="centered")

st.title("🔍 Local RAG Chat")
st.caption("Powered by Ollama + Chroma + sentence-transformers ")

# --- Chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# --- Display past messages ---
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.caption(f"📄 Sources: {', '.join(msg['sources'])}")

# --- Input ---
if user_input := st.chat_input("Ask something about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Get answer
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            result = ask(user_input)
        st.markdown(result["answer"])
        if result["sources"]:
            st.caption(f"📄 Sources: {', '.join(result['sources'])}")

    # Save assistant message
    st.session_state.messages.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
    })