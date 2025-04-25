import streamlit as st
import os
import tempfile
from werkzeug.utils import secure_filename
import time
from PIL import Image
import fitz  # PyMuPDF for PDF handling
import easyocr
import ssl
from io import BytesIO
import requests
from transformers import pipeline

# Import enhanced RAG system
from rag_system.enhanced_rag_system import EnhancedRAGSystem

# SSL configuration for EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

# Streamlit page config
st.set_page_config(page_title="RAG-SWAT with Web Search", layout="wide")
st.title("📘 RAG-SWAT with Web Search")

# Create temp folder for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize the enhanced RAG system
@st.cache_resource
def initialize_rag_system():
    api_key = "AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"  # Your Gemini API key
    rag = EnhancedRAGSystem(
        api_key=api_key,
        model_name="gemini-2.0-flash",
        embedding_model="all-MiniLM-L6-v2",
        confidence_threshold=0.  # Adjust this threshold as needed
    )
    return rag

# Initialize OCR reader
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

# Initialize LLaVA pipeline for image description
@st.cache_resource
def get_vlm_pipeline():
    try:
        vlm = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
        return vlm
    except Exception as e:
        st.error(f"Failed to load VLM: {e}")
        return None

# Initialize session state
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
if 'web_search_mode' not in st.session_state:
    st.session_state.web_search_mode = "auto"  # Options: "auto", "always", "never"

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
    """Generate a very short summary using Gemini"""
    if not text or len(text.strip()) < 10:
        return "Not enough content to generate summary"
    
    try:
        # Use the LLM from the RAG system to generate summary
        rag = initialize_rag_system()
        
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
        
        # Direct LLM call for summarization
        response = rag.llm.invoke(prompt)
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
    """Generate a summary of the extracted text"""
    if not text or len(text.strip()) < 100:
        return text

    try:
        rag = initialize_rag_system()
        prompt = f"""
        Please summarize the following text extracted from the document titled '{title}'. 
        Maintain key facts, figures, and important information:
        
        {text[:10000]}
        """
        response = rag.llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else response
        
        return f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"
    except Exception as e:
        st.error(f"Summarization Error: {e}")
        return text

def process_pdf(file_path, filename):
    """Enhanced PDF processing with OCR fallback"""
    try:
        doc = fitz.open(file_path)
        pdf_text = ""
        needs_ocr = False
        
        # First attempt: try to extract text normally
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():  # If we got some text
                pdf_text += page_text
            else:
                needs_ocr = True
                break
        
        doc.close()
        
        # If we detected pages needing OCR or got very little text
        if needs_ocr or len(pdf_text.strip()) < 100:
            return process_pdf_with_ocr(file_path, filename)
        
        return pdf_text, False
    
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return process_pdf_with_ocr(file_path, filename)

def process_pdf_with_ocr(file_path, filename):
    """Process PDF using OCR for image-based pages"""
    try:
        doc = fitz.open(file_path)
        ocr_text = ""
        reader = get_ocr_reader()
        
        for page_num in range(len(doc)):
            page = doc.load_page(page_num)
            pix = page.get_pixmap()
            
            # Convert pixmap to bytes for OCR
            img_bytes = pix.tobytes("png")
            
            # Save temporary image for OCR
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                temp.write(img_bytes)
                temp_path = temp.name
            
            # Perform OCR
            result = reader.readtext(temp_path)
            page_text = "\n".join([entry[1] for entry in result])
            ocr_text += f"Page {page_num+1}:\n{page_text}\n\n"
            
            # Clean up
            os.unlink(temp_path)
        
        doc.close()
        return ocr_text, True
    
    except Exception as e:
        st.error(f"OCR processing failed: {e}")
        return "", False

def process_files(uploaded_files):
    saved_paths = []
    processed_files = []
    summaries = []
    
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
                # Process PDF with enhanced handling
                pdf_text, used_ocr = process_pdf(file_path, filename)
                
                if pdf_text:
                    if used_ocr:
                        text_filename = f"{filename}_OCR.txt"
                        text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        
                        with open(text_path, 'w', encoding='utf-8') as text_file:
                            text_file.write(pdf_text)
                        
                        saved_paths.append(text_path)
                        st.session_state.document_names.append(f"{filename} (PDF via OCR)")
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": text_filename,
                            "type": "pdf_ocr"
                        })
                    else:
                        saved_paths.append(file_path)
                        st.session_state.document_names.append(filename)
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": filename,
                            "type": "document"
                        })
                    
                    summary = generate_short_summary(pdf_text, filename)
                else:
                    st.warning(f"Could not extract text from PDF: {filename}")
                    continue
            
            # For other document types (txt, docx, etc.)
            else:
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
            
            # Add summary to summaries list if available
            if summary:
                summaries.append({
                    "filename": filename,
                    "summary": summary,
                    "type": "image" if is_image else "document"
                })
    
    return saved_paths, processed_files, summaries

def display_chat_message(role, content, sources=None, is_web_search=False):
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
                        
                        # Choose appropriate icon based on source
                        if "Web:" in doc_name:
                            icon = "🌐"
                        elif "(OCR)" in doc_name:
                            icon = "🖼️ (Text from Image)"
                        elif "(Image Description)" in doc_name:
                            icon = "🖼️ (Image Description)"
                        else:
                            icon = "📄"
                        
                        # For web sources, display URL if available
                        if "Web:" in doc_name and "source_url" in source['metadata']:
                            st.markdown(f"**{icon} Source {i}:** *{doc_name}* - [Link]({source['metadata']['source_url']})")
                        else:
                            st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
                        content = source['content']
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.code(content)
            
            # Show an indication if result came from web search
            if is_web_search:
                st.info("ℹ️ This answer was generated from web search results because I couldn't find enough information in your documents.")

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
    
    # Web search mode selection
    st.subheader("🌐 Web Search Settings")
    web_search_mode = st.radio(
        "When to use web search:",
        options=["Auto (when documents don't have answers)", "Always", "Never"],
        index=0,
        key="web_search_radio"
    )
    
    # Map radio selection to state
    if web_search_mode == "Auto (when documents don't have answers)":
        st.session_state.web_search_mode = "auto"
    elif web_search_mode == "Always":
        st.session_state.web_search_mode = "always"
    else:
        st.session_state.web_search_mode = "never"
    
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
st.subheader("💬 Chat with Your Documents + Web")

# Display chat history
for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
    display_chat_message(
        chat["role"], 
        chat["content"], 
        chat.get("sources"),
        chat.get("from_web", False)
    )

# Chat input
user_question = st.chat_input("Ask a question about your documents or anything else...")

if user_question:
    # Add user question to chat history and display
    st.session_state.chat_history.append({"role": "user", "content": user_question})
    display_chat_message("user", user_question)
    
    # Determine if we should force web search
    force_web_search = st.session_state.web_search_mode == "always"
    skip_web_search = st.session_state.web_search_mode == "never"
    
    # Get and display assistant response
    with st.spinner("Thinking..."):
        try:
            start_time = time.time()
            
            if skip_web_search:
                # Only use RAG, even if it has low confidence
                result = rag._execute_rag_query(user_question)
                if "error" in result:
                    # If RAG fails, show a helpful message
                    result = {
                        "answer": "I couldn't find an answer in your documents. You might want to try enabling web search to expand my knowledge.",
                        "sources": [],
                        "query_time": time.time() - start_time,
                        "from_web": False
                    }
            else:
                # Use the enhanced query function with auto web search fallback
                result = rag.query(user_question, force_web_search=force_web_search)
            
            query_time = round(time.time() - start_time, 2)
            
            # Add assistant response to chat history
            chat_entry = {
                "role": "assistant",
                "content": result['answer'],
                "sources": result.get('sources', []),
                "from_web": result.get('from_web', False),
                "query_time": query_time
            }
            st.session_state.chat_history.append(chat_entry)
            
            # Display the response
            display_chat_message(
                "assistant", 
                result['answer'], 
                result.get('sources', []),
                result.get('from_web', False)
            )
            
            # Show response time with appropriate icon
            if result.get('from_web', False):
                st.success(f"⏱️ Response time: {query_time} seconds (🌐 Web Search)")
            else:
                st.success(f"⏱️ Response time: {query_time} seconds (📚 Documents)")
                
        except Exception as e:
            st.error(f"Error processing query: {e}")

