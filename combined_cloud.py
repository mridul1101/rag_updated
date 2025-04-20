import streamlit as st
import os
import tempfile
from werkzeug.utils import secure_filename
import time
from PIL import Image
from langchain_google_genai import ChatGoogleGenerativeAI
import fitz  # PyMuPDF for PDF handling
import easyocr
import ssl
from io import BytesIO
import requests
from transformers import pipeline

# Import RAG system components
from rag_system.rag_system import RAGSystem

# SSL configuration for EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

# Streamlit page config
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📘 RAG-SWAT")

# Create temp folder for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize the RAG system
@st.cache_resource
def initialize_rag_system():
    rag = RAGSystem(
        api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk",
        model_name="gemini-2.0-flash",
        embedding_model="all-MiniLM-L6-v2"
    )
    return rag

# Initialize Gemini for text processing and summarization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"
)

# Initialize EasyOCR reader
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

# Initialize LLaVA pipeline for image description
@st.cache_resource
def get_vlm_pipeline():
    try:
        # Using a smaller model for faster inference
        vlm = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        return vlm
    except Exception as e:
        st.error(f"Failed to load VLM: {e}")
        return None

# Initialize session state for documents and chat history
if 'all_documents' not in st.session_state:
    st.session_state.all_documents = []
if 'document_names' not in st.session_state:
    st.session_state.document_names = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_text_from_image(image_bytes):
    """Extract text from image using EasyOCR"""
    try:
        reader = get_ocr_reader()
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(image_bytes)
            temp_path = temp.name
        
        result = reader.readtext(temp_path)
        text = "\n".join([entry[1] for entry in result])
        
        os.unlink(temp_path)
        return text
    except Exception as e:
        st.error(f"OCR Error: {e}")
        return ""

def describe_image(image_bytes):
    """Generate description of image using VLM"""
    try:
        vlm = get_vlm_pipeline()
        if not vlm:
            return "Unable to load image description model"
            
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            temp.write(image_bytes)
            temp_path = temp.name
        
        # Open the image
        image = Image.open(temp_path)
        
        # Generate description
        description = vlm(image)[0]['generated_text']
        
        os.unlink(temp_path)
        return description
    except Exception as e:
        st.error(f"Image description error: {e}")
        return ""

def generate_short_summary(text, title, is_image=False):
    """Generate a very short summary (max 100 words) using Gemini"""
    if not text or len(text.strip()) < 10:
        return "Not enough content to generate summary"
    
    try:
        if is_image:
            prompt = f"""
            Please provide a very short summary (maximum 100 words) of this image titled '{title}' 
            based on its visual description. Focus on the main subject and key visual elements:
            
            {text[:5000]}
            """
        else:
            prompt = f"""
            Please provide a very short summary (maximum 100 words) of the following content from '{title}'. 
            Focus on the main topic and key points:
            
            {text[:5000]}
            """
        
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else response
        
        # Ensure summary is within word limit
        words = summary.split()
        if len(words) > 100:
            summary = ' '.join(words[:100]) + "..."
        
        return summary
    except Exception as e:
        st.error(f"Short summary error: {e}")
        return "Summary unavailable"

def summarize_text(text, title):
    """Generate a summary of the extracted text using Gemini via LangChain"""
    if not text or len(text.strip()) < 100:
        return text

    try:
        prompt = f"""
        Please summarize the following text extracted from the document titled '{title}'. 
        Maintain key facts, figures, and important information:
        
        {text[:10000]}
        """
        response = llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else response
        
        return f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"
    except Exception as e:
        st.error(f"Summarization Error: {e}")
        return text

def process_files(uploaded_files):
    saved_paths = []
    processed_files = []
    summaries = []  # To store summaries for display
    
    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.name):
            filename = secure_filename(uploaded_file.name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            summary = None
            is_image = file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']
            
            if is_image:
                # Process image files
                extracted_text = extract_text_from_image(uploaded_file.getvalue())
                
                if not extracted_text or len(extracted_text.strip()) < 10:
                    # If no text found, use VLM to describe the image
                    description = describe_image(uploaded_file.getvalue())
                    
                    if description:
                        text_filename = f"{filename}_description.txt"
                        text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        
                        with open(text_path, 'w', encoding='utf-8') as text_file:
                            text_file.write(f"IMAGE DESCRIPTION: {description}")
                        
                        saved_paths.append(text_path)
                        st.session_state.document_names.append(f"{filename} (Image Description)")
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": text_filename,
                            "type": "image_description"
                        })
                        summary = generate_short_summary(description, filename, is_image=True)
                else:
                    # We have OCR text
                    text_filename = f"{filename}_ocr.txt"
                    text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                    
                    processed_text = summarize_text(extracted_text, filename)
                    
                    with open(text_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(processed_text)
                    
                    saved_paths.append(text_path)
                    st.session_state.document_names.append(f"{filename} (OCR)")
                    processed_files.append({
                        "original_name": filename,
                        "processed_name": text_filename,
                        "type": "image_ocr"
                    })
                    summary = generate_short_summary(extracted_text, filename, is_image=True)
            
            elif file_ext == 'pdf':
                # Process PDF files
                try:
                    doc = fitz.open(file_path)
                    pdf_text = ""
                    for page in doc:
                        pdf_text += page.get_text()
                    doc.close()
                    
                    if pdf_text:
                        summary = generate_short_summary(pdf_text, filename)
                except Exception as e:
                    st.error(f"Error reading PDF: {e}")
                
                saved_paths.append(file_path)
                st.session_state.document_names.append(filename)
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
            
            else:
                # Process other document types
                saved_paths.append(file_path)
                st.session_state.document_names.append(filename)
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
                
                # Try to read text files for summary
                if file_ext == 'txt':
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            text = f.read(5000)  # Read first 5000 chars
                            summary = generate_short_summary(text, filename)
                    except:
                        pass
            
            if summary:
                summaries.append({
                    "filename": filename,
                    "summary": summary,
                    "type": "image" if is_image else "document"
                })
    
    return saved_paths, processed_files, summaries

def display_chat_message(role, content, sources=None):
    """Display a chat message with appropriate styling"""
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(f"**{content}**")
            
            if sources:
                with st.expander("View Sources", expanded=False):
                    for i, source in enumerate(sources, 1):
                        doc_name = source['metadata'].get('document_name', 'Unknown')
                        
                        if "(OCR)" in doc_name:
                            icon = "🖼️ (Text from Image)"
                        elif "(Image Description)" in doc_name:
                            icon = "🖼️ (Image Description)"
                        else:
                            icon = "📄"
                        
                        st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
                        content = source['content']
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.code(content)

# Get RAG instance
rag = initialize_rag_system()

# Sidebar for document upload and management
with st.sidebar:
    st.subheader("📂 Document Management")
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Upload documents", 
        type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "bmp", "tiff"], 
        accept_multiple_files=True,
        key="file_uploader"
    )
    
    # Process button
    if st.button("Process Documents", key="process_btn"):
        if uploaded_files:
            with st.spinner("Processing documents..."):
                try:
                    saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
                    # Display summaries immediately
                    if summaries:
                        st.subheader("📝 Quick Summaries")
                        for item in summaries:
                            icon = "🖼️" if item["type"] == "image" else "📄"
                            with st.expander(f"{icon} {item['filename']}", expanded=False):
                                st.write(item["summary"])
                    
                    new_documents = rag.document_processor.process_documents(saved_paths)
                    st.session_state.all_documents.extend(new_documents)
                    rag.create_vector_store(st.session_state.all_documents)
                    rag.create_rag_chain(chain_type="stuff", k=4)
                    st.session_state.documents_processed = True
                    st.session_state.processed_files.extend(processed_files)
                    st.success(f"✅ Processed {len(uploaded_files)} files!")
                except Exception as e:
                    st.error(f"Error processing documents: {e}")
        else:
            st.warning("Please upload files first")
    
    # Display processed documents
    if st.session_state.get("documents_processed", False):
        st.subheader("📚 Processed Documents")
        standard_docs = []
        image_ocr_docs = []
        image_desc_docs = []
        
        for doc in st.session_state.get("processed_files", []):
            if doc.get("type") == "image_ocr":
                image_ocr_docs.append(doc)
            elif doc.get("type") == "image_description":
                image_desc_docs.append(doc)
            else:
                standard_docs.append(doc)
        
        if standard_docs:
            st.write("**Standard Documents:**")
            for doc in standard_docs:
                st.write(f"📄 {doc['original_name']}")
        
        if image_ocr_docs:
            st.write("**Images with Text (OCR):**")
            for doc in image_ocr_docs:
                st.write(f"🖼️ {doc['original_name']}")
        
        if image_desc_docs:
            st.write("**Images with Descriptions:**")
            for doc in image_desc_docs:
                st.write(f"🎨 {doc['original_name']}")
        
        st.write(f"**Total documents:** {len(standard_docs) + len(image_ocr_docs) + len(image_desc_docs)}")
    
    # Clear chat history button
    if st.button("Clear Chat History"):
        st.session_state.chat_history = []
        st.rerun()

# Main chat interface
st.subheader("💬 Chat with Your Documents")

# Display chat history
for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
    display_chat_message(chat["role"], chat["content"], chat.get("sources"))

# Chat input
user_question = st.chat_input("Ask a question about your documents...")

if user_question:
    # Add user question to chat history and display
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    display_chat_message("user", user_question)
    
    # Get and display assistant response
    with st.spinner("..."):
        try:
            start_time = time.time()
            result = rag.query(user_question)
            query_time = round(time.time() - start_time, 2)
            
            # Add assistant response to chat history
            st.session_state.chat_history.append({
                "role": "assistant",
                "content": result['answer'],
                "sources": result['sources']
            })
            
            # Display the response
            display_chat_message("assistant", result['answer'], result['sources'])
            st.success(f"⏱️ Response time: {query_time} seconds")
        except Exception as e:
            st.error(f"Error processing query: {e}")