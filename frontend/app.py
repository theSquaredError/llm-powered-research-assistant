import streamlit as st
import requests

API_URL = "http://localhost:8000/api/ask"
FEEDBACK_URL = "http://localhost:8000/api/feedback"

st.set_page_config(page_title="LLM Research Assistant")
st.title("🧠 LLM-Powered Research Assistant")

query = st.text_area("Enter your research question:")

if st.button("Get Answer") and query:
    with st.spinner("Retrieving answer..."):
        response = requests.post(API_URL, json={"query": query})
        if response.status_code == 200:
            answer = response.json().get("response")
            st.success("Answer:")
            st.write(answer)

            feedback = st.radio("Was this answer helpful?", ("Yes", "No"))
            if st.button("Submit Feedback"):
                feedback_data = {
                    "query": query,
                    "response": answer,
                    "user_feedback": "positive" if feedback == "Yes" else "negative"
                }
                requests.post(FEEDBACK_URL, json=feedback_data)
                st.success("✅ Feedback submitted. Thank you!")
        else:
            st.error("Failed to retrieve answer. Please check your backend server.")