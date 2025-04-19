# import streamlit as st
# import os
# import tempfile
# from werkzeug.utils import secure_filename
# import time
# from PIL import Image
# # import google.generativeai as genai
# from langchain_google_genai import ChatGoogleGenerativeAI
# import fitz  # PyMuPDF for PDF handling
# import easyocr
# import ssl
# from io import BytesIO

# # Import RAG system components (these would need to be in the same directory)
# # Assuming you have these modules as shown in your original code
# from rag_system.rag_system import RAGSystem

# # SSL configuration for EasyOCR
# ssl._create_default_https_context = ssl._create_unverified_context

# # Streamlit page config
# st.set_page_config(page_title="RAG Chat", layout="wide")
# st.title("📘 RAG-based Document QA System")

# # Create temp folder for uploaded files
# UPLOAD_FOLDER = tempfile.mkdtemp()
# ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# # Initialize the RAG system
# @st.cache_resource
# def initialize_rag_system():
#     rag = RAGSystem(
#         api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk",
#         model_name="gemini-2.0-flash",
#         embedding_model="all-MiniLM-L6-v2"
#     )
#     return rag

# # Initialize Gemini for image processing and summarization
# # genai.configure(api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk")
# # model = genai.GenerativeModel('gemini-2.0-flash')
# llm = ChatGoogleGenerativeAI(
#     model="gemini-2.0-flash",
#     temperature=0.7,
#     google_api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"
# )


# # Initialize EasyOCR reader
# @st.cache_resource
# def get_ocr_reader():
#     return easyocr.Reader(['en'])

# # Initialize session state for documents
# if 'all_documents' not in st.session_state:
#     st.session_state.all_documents = []
# if 'document_names' not in st.session_state:
#     st.session_state.document_names = []
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = []

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_image(image_bytes):
#     """Extract text from image using EasyOCR"""
#     try:
#         reader = get_ocr_reader()
#         # Create a temporary file to save the image
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
#             temp.write(image_bytes)
#             temp_path = temp.name
        
#         # Read text from the temp file
#         result = reader.readtext(temp_path)
#         text = "\n".join([entry[1] for entry in result])
        
#         # Clean up the temp file
#         os.unlink(temp_path)
#         return text
#     except Exception as e:
#         st.error(f"OCR Error: {e}")
#         return ""

# # def summarize_text(text, title):
# #     """Generate a summary of the extracted text using Gemini"""
# #     if not text or len(text.strip()) < 100:  # Skip summarization for very short texts
# #         return text
    
# #     try:
# #         prompt = f"""
# #         Please summarize the following text extracted from the document titled '{title}'. 
# #         Maintain key facts, figures, and important information:
        
# #         {text[:10000]}  # Limiting to first 10000 chars to avoid token limits
# #         """
        
# #         response = model.generate_content(prompt)
# #         summary = response.text
        
# #         # Combine summary with original text to maintain searchability
# #         result = f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"  # Include partial original text
# #         return result
# #     except Exception as e:
# #         st.error(f"Summarization Error: {e}")
# #         return text

# def summarize_text(text, title):
#     """Generate a summary of the extracted text using Gemini via LangChain"""
#     if not text or len(text.strip()) < 100:
#         return text

#     try:
#         prompt = f"""
#         Please summarize the following text extracted from the document titled '{title}'. 
#         Maintain key facts, figures, and important information:
        
#         {text[:10000]}
#         """
#         response = llm.invoke(prompt)
#         summary = response.content if hasattr(response, "content") else response
        
#         return f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"
#     except Exception as e:
#         st.error(f"Summarization Error: {e}")
#         return text
        
# def process_files(uploaded_files):
#     saved_paths = []
#     processed_files = []
    
#     for uploaded_file in uploaded_files:
#         if uploaded_file and allowed_file(uploaded_file.name):
#             filename = secure_filename(uploaded_file.name)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
            
#             # Save the uploaded file
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getvalue())
            
#             file_ext = filename.rsplit('.', 1)[1].lower()
            
#             # Process images with OCR and summarization
#             if file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
#                 extracted_text = extract_text_from_image(uploaded_file.getvalue())
#                 if extracted_text:
#                     # Create a text file with the extracted content
#                     text_filename = f"{filename}_ocr.txt"
#                     text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                    
#                     # Summarize extracted text
#                     processed_text = summarize_text(extracted_text, filename)
                    
#                     with open(text_path, 'w', encoding='utf-8') as text_file:
#                         text_file.write(processed_text)
                    
#                     saved_paths.append(text_path)
#                     st.session_state.document_names.append(f"{filename} (OCR)")
#                     processed_files.append({
#                         "original_name": filename,
#                         "processed_name": text_filename,
#                         "type": "image_ocr"
#                     })
#             else:
#                 # Standard document processing
#                 saved_paths.append(file_path)
#                 st.session_state.document_names.append(filename)
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": filename,
#                     "type": "document"
#                 })
    
#     return saved_paths, processed_files

# # Get RAG instance
# rag = initialize_rag_system()

# # File uploader section
# st.subheader("Upload Documents")
# uploaded_files = st.file_uploader(
#     "Upload documents to query", 
#     type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "bmp", "tiff"], 
#     accept_multiple_files=True
# )

# # If files are uploaded, process them
# if uploaded_files:
#     col1, col2 = st.columns([1, 3])
#     with col1:
#         process_button = st.button("Process Documents")
    
#     with col2:
#         if process_button:
#             with st.spinner("Processing documents and extracting text from images..."):
#                 try:
#                     saved_paths, processed_files = process_files(uploaded_files)
                    
#                     # Process the documents with the RAG system
#                     new_documents = rag.document_processor.process_documents(saved_paths)
                    
#                     # Add to all documents
#                     st.session_state.all_documents.extend(new_documents)
                    
#                     # Create or update vector store
#                     rag.create_vector_store(st.session_state.all_documents)
                    
#                     # Create RAG chain
#                     rag.create_rag_chain(chain_type="stuff", k=4)
                    
#                     st.session_state.documents_processed = True
#                     st.session_state.processed_files.extend(processed_files)
                    
#                     # Show success message with document count
#                     st.success(f"✅ Successfully processed {len(uploaded_files)} files!")
#                 except Exception as e:
#                     st.error(f"Error processing documents: {e}")

# # Display processed documents if available
# if st.session_state.get("documents_processed", False):
#     with st.expander("Processed Documents", expanded=True):
#         st.subheader("Available Documents")
        
#         # Group by document type
#         standard_docs = []
#         image_docs = []
        
#         for doc in st.session_state.get("processed_files", []):
#             if doc.get("type") == "image_ocr":
#                 image_docs.append(doc)
#             else:
#                 standard_docs.append(doc)
        
#         if standard_docs:
#             st.write("📄 **Standard Documents:**")
#             for i, doc in enumerate(standard_docs, 1):
#                 st.write(f"{i}. {doc['original_name']}")
        
#         if image_docs:
#             st.write("🖼️ **Images with Extracted Text:**")
#             for i, doc in enumerate(image_docs, 1):
#                 st.write(f"{i}. {doc['original_name']} → {doc['processed_name']}")
        
#         st.write(f"Total documents in knowledge base: {len(standard_docs) + len(image_docs)}")

# # Divider
# st.markdown("---")

# # Text input for the user's question
# st.subheader("Ask Questions")
# user_question = st.text_input("Ask a question from the uploaded documents:")

# # Submit button
# if st.button("Submit Query"):
#     if user_question:
#         with st.spinner("Getting answer..."):
#             try:
#                 # Start timing
#                 start_time = time.time()
                
#                 # Query the RAG system
#                 result = rag.query(user_question)
                
#                 # Calculate query time
#                 query_time = round(time.time() - start_time, 2)
                
#                 # Display answer
#                 st.markdown("### Answer:")
#                 st.markdown(f"🧠 **{result['answer']}**")
                
#                 # Display sources
#                 st.markdown("### 📄 Source Documents:")
#                 for i, source in enumerate(result["sources"], 1):
#                     doc_name = source['metadata'].get('document_name', 'Unknown')
                    
#                     # Check if it's from an image
#                     is_image = "(OCR)" in doc_name
#                     icon = "🖼️" if is_image else "📄"
                    
#                     st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                    
#                     # Format content with smaller font for better readability
#                     content = source['content']
#                     if len(content) > 500:
#                         content = content[:500] + "..."
#                     st.code(content)
                
#                 # Query time info
#                 st.success(f"⏱️ Query Time: {query_time} seconds")
#             except Exception as e:
#                 st.error(f"Error processing query: {e}")
#     else:
#         st.warning("Please enter a question.")



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

# Import RAG system components
from rag_system.rag_system import RAGSystem

# SSL configuration for EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

# Streamlit page config
st.set_page_config(page_title="RAG Chat", layout="wide")
st.title("📘 RAG-based Document QA System")

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

# Initialize Gemini for image processing and summarization
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    temperature=0.7,
    google_api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"
)

# Initialize EasyOCR reader
@st.cache_resource
def get_ocr_reader():
    return easyocr.Reader(['en'])

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
    
    for uploaded_file in uploaded_files:
        if uploaded_file and allowed_file(uploaded_file.name):
            filename = secure_filename(uploaded_file.name)
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            if file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                extracted_text = extract_text_from_image(uploaded_file.getvalue())
                if extracted_text:
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
            else:
                saved_paths.append(file_path)
                st.session_state.document_names.append(filename)
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
    
    return saved_paths, processed_files

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
                        is_image = "(OCR)" in doc_name
                        icon = "🖼️" if is_image else "📄"
                        
                        st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
                        content = source['content']
                        if len(content) > 500:
                            content = content[:500] + "..."
                        st.code(content)

# def display_chat_message(role, content, sources=None):
#     """Display a chat message with appropriate styling"""
#     if role == "user":
#         with st.chat_message("user", avatar="🧑"):
#             st.markdown(content)
#     else:
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(f"**{content}**")
            
#             if sources:
#                 # Display only unique source document names
#                 unique_sources = {}
#                 for source in sources:
#                     doc_name = source['metadata'].get('document_name', 'Unknown')
#                     if doc_name not in unique_sources:
#                         unique_sources[doc_name] = source
                
#                 st.markdown("---")
#                 st.markdown("**📚 Source Documents:**")
#                 for i, (doc_name, source) in enumerate(unique_sources.items(), 1):
#                     is_image = "(OCR)" in doc_name
#                     icon = "🖼️" if is_image else "📄"
#                     st.markdown(f"{icon} **{doc_name}**")
                
#                 # Keep the expandable section with full content
#                 with st.expander("View Detailed Source Content", expanded=False):
#                     for i, source in enumerate(sources, 1):
#                         doc_name = source['metadata'].get('document_name', 'Unknown')
#                         is_image = "(OCR)" in doc_name
#                         icon = "🖼️" if is_image else "📄"
                        
#                         st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
#                         content = source['content']
#                         if len(content) > 500:
#                             content = content[:500] + "..."
#                         st.code(content)
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
                    saved_paths, processed_files = process_files(uploaded_files)
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
        image_docs = []
        
        for doc in st.session_state.get("processed_files", []):
            if doc.get("type") == "image_ocr":
                image_docs.append(doc)
            else:
                standard_docs.append(doc)
        
        if standard_docs:
            st.write("**Standard Documents:**")
            for doc in standard_docs:
                st.write(f"📄 {doc['original_name']}")
        
        if image_docs:
            st.write("**Images with OCR:**")
            for doc in image_docs:
                st.write(f"🖼️ {doc['original_name']}")
        
        st.write(f"**Total documents:** {len(standard_docs) + len(image_docs)}")
    
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
    with st.spinner("Thinking..."):
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

