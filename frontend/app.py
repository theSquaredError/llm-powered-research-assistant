import streamlit as st
import requests
import json

st.set_page_config(page_title="LLM Research Assistant", layout="wide")
st.title("🤖 LLM-Powered Research Assistant")

API_URL = "http://localhost:8000/api/ask"
FEEDBACK_URL = "http://localhost:8000/api/feedback"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "feedback_submitted" not in st.session_state:
    st.session_state.feedback_submitted = False

user_query = st.text_input(
    "Ask a question about your research paper:",
    placeholder="e.g. What is the main contribution of this paper?"
)

if st.button("Ask") and user_query:
    st.session_state.feedback_submitted = False
    with st.spinner("Thinking..."):
        chat_history = st.session_state.chat_history.copy()
        payload = {
            "query": user_query,
            "history": chat_history
        }
        response = requests.post(API_URL, json=payload, stream=True)

        # response = requests.post(API_URL, json={"query": user_query}, stream=True)
        full_answer = ""
        response_placeholder = st.empty()
        for chunk in response.iter_lines():
            if chunk:
                try:
                    decoded_chunk = chunk.decode("utf-8").strip()
                    if decoded_chunk.startswith("data:"):
                        decoded_chunk = decoded_chunk[5:].strip()
                    if decoded_chunk.startswith("{"):
                        parsed = json.loads(decoded_chunk)
                        if isinstance(parsed, dict) and "response" in parsed:
                            full_answer += "".join(parsed["response"])
                    else:
                        full_answer += decoded_chunk
                    response_placeholder.markdown(full_answer.replace("\\n", "\n").replace(",", "").strip())
                except Exception as e:
                    print(f"[Error] Decoding chunk: {e}")
        st.session_state.chat_history.append((user_query, full_answer))

st.markdown("---")
st.subheader("💬 Chat History")
for i, (question, answer) in enumerate(st.session_state.chat_history):
    with st.container():
        st.markdown(f"**👤 You:** {question}")
        st.markdown(f"**🤖 Assistant:** {answer}")
        st.markdown("---")

if st.session_state.chat_history:
    last_query, last_response = st.session_state.chat_history[-1]
    if not st.session_state.feedback_submitted:
        st.markdown("#### Was this helpful?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Yes"):
                requests.post(FEEDBACK_URL, json={
                    "query": last_query,
                    "response": last_response,
                    "user_feedback": "yes"
                })
                st.session_state.feedback_submitted = True
        with col2:
            if st.button("👎 No"):
                requests.post(FEEDBACK_URL, json={
                    "query": last_query,
                    "response": last_response,
                    "user_feedback": "no"
                })
                st.session_state.feedback_submitted = True
    else:
        st.success("✅ Thanks for your feedback!")
