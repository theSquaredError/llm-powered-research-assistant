import sys
from pathlib import Path
project_root = Path(__file__).parent.parent.resolve()
sys.path.insert(0, str(project_root))
import os
import streamlit as st
import pandas as pd
from pathlib import Path
import sys
from streamlit_extras.bottom_container import bottom
import requests
from scripts.ingest3 import DocumentProcessor, QdrantIndexer
# load_dotenv()

API_URL = "http://localhost:8000/api/ask"
FEEDBACK_URL = "http://localhost:8000/api/feedback"
st.set_page_config(
    page_title = "Research Assistant",
    layout="wide"
)
st.title("Research Assistant")

def process_and_index():
    processor = DocumentProcessor()
    indexer = QdrantIndexer(collection_name="papers")
    
    # clear old data before indexing new documents
    indexer.clear_collection()

    full_markdown = ""
    docs_to_index = []

    # Process uploaded files
    if st.session_state.uploaded_files:
        md, docs = processor.process_uploaded_files(st.session_state.uploaded_files)
        full_markdown += md + "\n"
        docs_to_index.extend(docs)

    # Process URLs
    if st.session_state.get("file_urls"):
        md, docs = processor.process_urls(st.session_state.file_urls)
        full_markdown += md + "\n"
        docs_to_index.extend(docs)

    for doc in docs_to_index:
        indexer.index_document(
            markdown_text=doc['markdown'],
            doc_obj=doc['doc'],
            source_name=doc['filename']
        )
    
    return full_markdown, indexer


def stream_response_from_api(query:str):
    """
    Generator that streams tokens from the FASTAPI backend
    Yields token one by one fore real-time display.
    """
    payload = {"query": query}
    try:
        response = requests.post(
            API_URL,
            json = payload,
            stream = True,
            timeout = 300
        )
        response.raise_for_status()

        # stream text chunks
        for chunk in response.iter_content(decode_unicode=True):
            if chunk:
                yield chunk
    except requests.exceptions.ConnectionError:
        yield "Error: Could not connect to API at " + API_URL
    except requests.exceptions.Timeout:
        yield "Error: API request timed out"
    except requests.exceptions.RequestException as e:
        yield f"Error calling API: {str(e)}"
    except Exception as e:
        yield f"Unexpected error: {str(e)}"

def initialise_session_state():
    if "uploaded_files" not in st.session_state:
        st.session_state.uploaded_files = []
    if "file_urls" not in st.session_state:
        st.session_state.file_urls = []
    if  "vectorstore" not in st.session_state:
        st.session_state.vectorstore = None
    if "agent" not in st.session_state:
        st.session_state.agent = None
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "processing_status" not in st.session_state:
        st.session_state.processing_status = "not_started"
    if "docling_docs" not in st.session_state:
        st.session_state.docling_docs = []

def render_sidebar():
    """Render the sidebar with setup controls"""

    with st.sidebar:
        st.title("Research Assistant")
        # new chat / clear history button
        col1, col2  = st.columns(2)
        with col1:
            if st.button(" New chat", use_container_width=True):
                st.session_state.messages = []
                st.success("Chat cleared!")
                st.rerun()

        with col2:
            if st.button("Delete All", use_container_width=True):
                st.session_state.messages = []
                st.session_state.uploaded_files = []
                st.session_state.file_urls = []
                st.session_state.vectorstore = None
                st.session_state.agent = None
                st.session_state.processing_status = "not_started"
                st.info("All data cleared!")
                st.rerun()

        st.divider()
        st.title("setup")
        # file uploader
        uploaded_files = st.file_uploader(
            "Upload Document",
            type=["pdf", "docx", "pptx", "html"],
            accept_multiple_files=True,
            help="Upload PDF, Word (DOCX), PowerPoint (PPTX), or HTML files",
        )

        if uploaded_files:
            st.info(f"{len(uploaded_files)} file(s) uploaded")

            # List uploaded files
            with st.expander("Uploade Files"):
                for file in uploaded_files:
                    st.write(f"- {file.name} ({file.type})")
        
        # URL Input
        url_input = st.text_area("Or provide URLs (one per line)", height=100, 
                                 help="Enter URLs of documents to process")

        # Process button
        if st.button("Process & Index", use_container_width=True):
            st.session_state.uploaded_files = uploaded_files
            st.session_state.file_urls = [url.strip() for url in url_input.split('\n') if url.strip()]
            
            if st.session_state.uploaded_files or st.session_state.file_urls:
                st.session_state.processing_status = "processing"
                with st.spinner("Processing documents..."):
                    try:
                        markdown, indexer = process_and_index()
                        st.session_state.vectorstore = indexer
                        st.session_state.processing_status = "completed"
                        st.session_state.agent = "Ready"
                    except Exception as e:
                        st.error(f"Error: {e}")
                        st.session_state.processing_status = "error"
            else:
                st.warning("Please upload files or provide URLs.")
            
        
        # Status indicator

        st.divider()
        st.subheader("Status")

        if st.session_state.processing_status == "not_started":
            st.info("Ready to start")
        elif st.session_state.processing_status == "completed":
            st.success("Ready to chat!")
        elif st.session_state.processing_status == "error":
            st.error("Error occurred")

        # Tips
        with st.expander("Tips"):
            st.markdown(
                """
            **Supported formats:**
            - PDF documents
            - Word documents (.docx)
            - PowerPoint presentations (.pptx)
            - HTML files

            **Best practises:**
            - Uploaded related documents together
            - Start with a few documents for testing
            - Documents are processed with OCR for scanned content
            - Tables and structures are preserved
            """
            # **For production:**
            # - Add persistent vector storage
            # - Implement GPU acceleration for faster processing
            # - Add authentication and access controls

            )

def render_chat():
    """Render the chat interface"""
    # check if the agent is ready
    if st.session_state.agent is None:
        st.info("Please upload and process your documents in the sidebar first!")
        st.markdown(
            """
        ### How to use:
        1. Upload your documents in the sidebar (PDF, DOCX, PPTX, or HTML)
        2. Click "Process & Index" and wait for processing
        3. Start asking questions about your documents!

        ### What you can do:
        - Ask questions about document content
        - Compare information across multiple documents
        - Extract specific data or insight
        - Summarise document section

        """
        )
        return
    
    # Display chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input in bottom container (attempt to fix positioning in tabs)
    with bottom():
        prompt = st.chat_input("Ask a question about your documents...")
    
    if prompt:
        # add user message
        st.session_state.messages.append({"role":"user","content":prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # get agent response
        with st.chat_message("assistant"):
            # create status and message placeholders
            status_placeholder = st.empty()
            message_placeholder = st.empty()

            try:
                status_placeholder.markdown("**Thinking...**")
                full_response = ""

                # Stream response token by token
                for token in stream_response_from_api(prompt):
                    full_response += token
                    # show cursor while streaming
                    message_placeholder.markdown(full_response +"| ")
                
                # final response without cursor
                message_placeholder.markdown(full_response)
                status_placeholder.empty()


            except Exception as e:
                import traceback
                error_details = traceback.format_exc()
                print(f"Chat error: {error_details}") # log to console
                error_msg = f"Error: {str(e)}"
                status_placeholder.empty()
                message_placeholder.markdown(error_msg)
                full_response = error_msg

        # Add assistant response to history
        st.session_state.messages.append(
            {"role":"assistant", "content": full_response}
        )

def main():
    initialise_session_state()
    render_sidebar()

    # create tabs
    tab1 = st.tabs(["Chat"])
    with tab1[0]:
        render_chat()

if __name__ == "__main__":
    main()