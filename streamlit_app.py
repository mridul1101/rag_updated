import streamlit as st
import requests
import json
import os

# Streamlit page config
st.set_page_config(page_title="RAG Chat", layout="wide")

st.title("📘 RAG-based Document QA System")

# Backend URLs
API_URL = "http://localhost:5000/query"
UPLOAD_URL = "http://localhost:5000/upload"

# File uploader section
st.subheader("Upload Documents")
uploaded_files = st.file_uploader(
    "Upload documents to query", 
    type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "bmp", "tiff"], 
    accept_multiple_files=True
)

# If files are uploaded, send them to the backend
if uploaded_files:
    col1, col2 = st.columns([1, 3])
    with col1:
        process_button = st.button("Process Documents")
    
    with col2:
        if process_button:
            with st.spinner("Processing documents and extracting text from images..."):
                files = []
                for uploaded_file in uploaded_files:
                    files.append(('files', (uploaded_file.name, uploaded_file.getvalue(), 
                                 'application/octet-stream')))
                
                try:
                    response = requests.post(UPLOAD_URL, files=files)
                    if response.status_code == 200:
                        doc_data = response.json()
                        st.session_state.documents_processed = True
                        st.session_state.processed_files = doc_data.get("processed_files", [])
                        
                        # Show success message with document count
                        st.success(f"✅ Successfully processed {len(uploaded_files)} files!")
                    else:
                        st.error(f"❌ Error processing documents: {response.text}")
                except Exception as e:
                    st.error(f"Error connecting to API: {e}")

# Display processed documents if available
if st.session_state.get("documents_processed", False):
    with st.expander("Processed Documents", expanded=True):
        st.subheader("Available Documents")
        
        # Group by document type
        standard_docs = []
        image_docs = []
        
        for doc in st.session_state.get("processed_files", []):
            if doc.get("type") == "image_ocr":
                image_docs.append(doc)
            else:
                standard_docs.append(doc)
        
        if standard_docs:
            st.write("📄 **Standard Documents:**")
            for i, doc in enumerate(standard_docs, 1):
                st.write(f"{i}. {doc['original_name']}")
        
        if image_docs:
            st.write("🖼️ **Images with Extracted Text:**")
            for i, doc in enumerate(image_docs, 1):
                st.write(f"{i}. {doc['original_name']} → {doc['processed_name']}")
        
        st.write(f"Total documents in knowledge base: {len(standard_docs) + len(image_docs)}")

# Divider
st.markdown("---")

# Text input for the user's question
st.subheader("Ask Questions")
user_question = st.text_input("Ask a question from the uploaded documents:")

# Submit button
if st.button("Submit Query"):
    if user_question:
        with st.spinner("Getting answer..."):
            try:
                # Send POST request to Flask API
                response = requests.post(API_URL, json={"question": user_question})
                
                if response.status_code == 200:
                    data = response.json()
                    
                    # Display answer
                    st.markdown("### Answer:")
                    st.markdown(f"🧠 **{data['answer']}**")
                    
                    # Display sources
                    st.markdown("### 📄 Source Documents:")
                    for i, source in enumerate(data["sources"], 1):
                        doc_name = source['document_name']
                        
                        # Check if it's from an image
                        is_image = "(OCR)" in doc_name
                        icon = "🖼️" if is_image else "📄"
                        
                        st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
                        # Format content with smaller font for better readability
                        content = source['content']
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.code(content)
                    
                    # Query time info
                    st.success(f"⏱️ Query Time: {data['query_time']} seconds")
                else:
                    st.error(f"❌ Error: {response.text}")
            except Exception as e:
                st.exception(f"Error connecting to API: {e}")
    else:
        st.warning("Please enter a question.")

# Initialize session state
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []