# import streamlit as st
# import os
# import tempfile
# from werkzeug.utils import secure_filename
# import time
# from PIL import Image
# import fitz  # PyMuPDF for PDF handling
# import easyocr
# import ssl
# from io import BytesIO
# import requests
# from transformers import pipeline

# # Import enhanced RAG system
# from rag_system.enhanced_rag_system import EnhancedRAGSystem

# # SSL configuration for EasyOCR
# ssl._create_default_https_context = ssl._create_unverified_context

# # Streamlit page config
# st.set_page_config(page_title="RAG-SWAT with Web Search", layout="wide")
# st.title("📘 RAG-SWAT with Web Search")

# # Create temp folder for uploaded files
# UPLOAD_FOLDER = tempfile.mkdtemp()
# ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# # Initialize the enhanced RAG system
# @st.cache_resource
# def initialize_rag_system():
#     api_key = "AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"  # Your Gemini API key
#     rag = EnhancedRAGSystem(
#         api_key=api_key,
#         model_name="gemini-2.0-flash",
#         embedding_model="all-MiniLM-L6-v2",
#         confidence_threshold=0.  # Adjust this threshold as needed
#     )
#     return rag

# # Initialize OCR reader
# @st.cache_resource
# def get_ocr_reader():
#     return easyocr.Reader(['en'])

# # Initialize LLaVA pipeline for image description
# @st.cache_resource
# def get_vlm_pipeline():
#     try:
#         vlm = pipeline("image-to-text", model="Salesforce/blip-image-captioning-base")
#         return vlm
#     except Exception as e:
#         st.error(f"Failed to load VLM: {e}")
#         return None

# # Initialize session state
# if 'all_documents' not in st.session_state:
#     st.session_state.all_documents = []
# if 'document_names' not in st.session_state:
#     st.session_state.document_names = []
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = []
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'web_search_mode' not in st.session_state:
#     st.session_state.web_search_mode = "auto"  # Options: "auto", "always", "never"

# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_image(image_bytes):
#     """Extract text from image using EasyOCR"""
#     try:
#         reader = get_ocr_reader()
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
#             temp.write(image_bytes)
#             temp_path = temp.name
        
#         result = reader.readtext(temp_path)
#         text = "\n".join([entry[1] for entry in result])
        
#         os.unlink(temp_path)
#         return text
#     except Exception as e:
#         st.error(f"OCR Error: {e}")
#         return ""

# def describe_image(image_bytes):
#     """Generate description of image using VLM"""
#     try:
#         vlm = get_vlm_pipeline()
#         if not vlm:
#             return "Unable to load image description model"
            
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
#             temp.write(image_bytes)
#             temp_path = temp.name
        
#         # Open the image
#         image = Image.open(temp_path)
        
#         # Generate description
#         description = vlm(image)[0]['generated_text']
        
#         os.unlink(temp_path)
#         return description
#     except Exception as e:
#         st.error(f"Image description error: {e}")
#         return ""

# def generate_short_summary(text, title, is_image=False):
#     """Generate a very short summary using Gemini"""
#     if not text or len(text.strip()) < 10:
#         return "Not enough content to generate summary"
    
#     try:
#         # Use the LLM from the RAG system to generate summary
#         rag = initialize_rag_system()
        
#         if is_image:
#             prompt = f"""
#             Please provide a very short summary (maximum 100 words) of this image titled '{title}' 
#             based on its visual description. Focus on the main subject and key visual elements:
            
#             {text[:5000]}
#             """
#         else:
#             prompt = f"""
#             Please provide a very short summary (maximum 100 words) of the following content from '{title}'. 
#             Focus on the main topic and key points:
            
#             {text[:5000]}
#             """
        
#         # Direct LLM call for summarization
#         response = rag.llm.invoke(prompt)
#         summary = response.content if hasattr(response, "content") else response
        
#         # Ensure summary is within word limit
#         words = summary.split()
#         if len(words) > 100:
#             summary = ' '.join(words[:100]) + "..."
        
#         return summary
#     except Exception as e:
#         st.error(f"Short summary error: {e}")
#         return "Summary unavailable"

# def summarize_text(text, title):
#     """Generate a summary of the extracted text"""
#     if not text or len(text.strip()) < 100:
#         return text

#     try:
#         rag = initialize_rag_system()
#         prompt = f"""
#         Please summarize the following text extracted from the document titled '{title}'. 
#         Maintain key facts, figures, and important information:
        
#         {text[:10000]}
#         """
#         response = rag.llm.invoke(prompt)
#         summary = response.content if hasattr(response, "content") else response
        
#         return f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"
#     except Exception as e:
#         st.error(f"Summarization Error: {e}")
#         return text

# def process_pdf(file_path, filename):
#     """Enhanced PDF processing with OCR fallback"""
#     try:
#         doc = fitz.open(file_path)
#         pdf_text = ""
#         needs_ocr = False
        
#         # First attempt: try to extract text normally
#         for page in doc:
#             page_text = page.get_text()
#             if page_text.strip():  # If we got some text
#                 pdf_text += page_text
#             else:
#                 needs_ocr = True
#                 break
        
#         doc.close()
        
#         # If we detected pages needing OCR or got very little text
#         if needs_ocr or len(pdf_text.strip()) < 100:
#             return process_pdf_with_ocr(file_path, filename)
        
#         return pdf_text, False
    
#     except Exception as e:
#         st.error(f"Error reading PDF: {e}")
#         return process_pdf_with_ocr(file_path, filename)

# def process_pdf_with_ocr(file_path, filename):
#     """Process PDF using OCR for image-based pages"""
#     try:
#         doc = fitz.open(file_path)
#         ocr_text = ""
#         reader = get_ocr_reader()
        
#         for page_num in range(len(doc)):
#             page = doc.load_page(page_num)
#             pix = page.get_pixmap()
            
#             # Convert pixmap to bytes for OCR
#             img_bytes = pix.tobytes("png")
            
#             # Save temporary image for OCR
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
#                 temp.write(img_bytes)
#                 temp_path = temp.name
            
#             # Perform OCR
#             result = reader.readtext(temp_path)
#             page_text = "\n".join([entry[1] for entry in result])
#             ocr_text += f"Page {page_num+1}:\n{page_text}\n\n"
            
#             # Clean up
#             os.unlink(temp_path)
        
#         doc.close()
#         return ocr_text, True
    
#     except Exception as e:
#         st.error(f"OCR processing failed: {e}")
#         return "", False

# def process_files(uploaded_files):
#     saved_paths = []
#     processed_files = []
#     summaries = []
    
#     for uploaded_file in uploaded_files:
#         if uploaded_file and allowed_file(uploaded_file.name):
#             filename = secure_filename(uploaded_file.name)
#             file_path = os.path.join(UPLOAD_FOLDER, filename)
            
#             with open(file_path, "wb") as f:
#                 f.write(uploaded_file.getvalue())
            
#             file_ext = filename.rsplit('.', 1)[1].lower()
#             summary = None
#             is_image = file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']

#             if is_image:
#                 # Process image files
#                 extracted_text = extract_text_from_image(uploaded_file.getvalue())
                
#                 if not extracted_text or len(extracted_text.strip()) < 10:
#                     # If no text found, use VLM to describe the image
#                     description = describe_image(uploaded_file.getvalue())
                    
#                     if description:
#                         text_filename = f"{filename}_description.txt"
#                         text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        
#                         with open(text_path, 'w', encoding='utf-8') as text_file:
#                             text_file.write(f"IMAGE DESCRIPTION: {description}")
                        
#                         saved_paths.append(text_path)
#                         st.session_state.document_names.append(f"{filename} (Image Description)")
#                         processed_files.append({
#                             "original_name": filename,
#                             "processed_name": text_filename,
#                             "type": "image_description"
#                         })
#                         summary = generate_short_summary(description, filename, is_image=True)
#                 else:
#                     # We have OCR text
#                     text_filename = f"{filename}_ocr.txt"
#                     text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                    
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
#                     summary = generate_short_summary(extracted_text, filename, is_image=True)
            
#             elif file_ext == 'pdf':
#                 # Process PDF with enhanced handling
#                 pdf_text, used_ocr = process_pdf(file_path, filename)
                
#                 if pdf_text:
#                     if used_ocr:
#                         text_filename = f"{filename}_OCR.txt"
#                         text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        
#                         with open(text_path, 'w', encoding='utf-8') as text_file:
#                             text_file.write(pdf_text)
                        
#                         saved_paths.append(text_path)
#                         st.session_state.document_names.append(f"{filename} (PDF via OCR)")
#                         processed_files.append({
#                             "original_name": filename,
#                             "processed_name": text_filename,
#                             "type": "pdf_ocr"
#                         })
#                     else:
#                         saved_paths.append(file_path)
#                         st.session_state.document_names.append(filename)
#                         processed_files.append({
#                             "original_name": filename,
#                             "processed_name": filename,
#                             "type": "document"
#                         })
                    
#                     summary = generate_short_summary(pdf_text, filename)
#                 else:
#                     st.warning(f"Could not extract text from PDF: {filename}")
#                     continue
            
#             # For other document types (txt, docx, etc.)
#             else:
#                 saved_paths.append(file_path)
#                 st.session_state.document_names.append(filename)
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": filename,
#                     "type": "document"
#                 })
                
#                 # Try to read text files for summary
#                 if file_ext == 'txt':
#                     try:
#                         with open(file_path, 'r', encoding='utf-8') as f:
#                             text = f.read(5000)  # Read first 5000 chars
#                             summary = generate_short_summary(text, filename)
#                     except:
#                         pass
            
#             # Add summary to summaries list if available
#             if summary:
#                 summaries.append({
#                     "filename": filename,
#                     "summary": summary,
#                     "type": "image" if is_image else "document"
#                 })
    
#     return saved_paths, processed_files, summaries

# def display_chat_message(role, content, sources=None, is_web_search=False):
#     """Display a chat message with appropriate styling"""
#     if role == "user":
#         with st.chat_message("user", avatar="🧑"):
#             st.markdown(content)
#     else:
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(f"**{content}**")
            
#             if sources:
#                 with st.expander("View Sources", expanded=False):
#                     for i, source in enumerate(sources, 1):
#                         doc_name = source['metadata'].get('document_name', 'Unknown')
                        
#                         # Choose appropriate icon based on source
#                         if "Web:" in doc_name:
#                             icon = "🌐"
#                         elif "(OCR)" in doc_name:
#                             icon = "🖼️ (Text from Image)"
#                         elif "(Image Description)" in doc_name:
#                             icon = "🖼️ (Image Description)"
#                         else:
#                             icon = "📄"
                        
#                         # For web sources, display URL if available
#                         if "Web:" in doc_name and "source_url" in source['metadata']:
#                             st.markdown(f"**{icon} Source {i}:** *{doc_name}* - [Link]({source['metadata']['source_url']})")
#                         else:
#                             st.markdown(f"**{icon} Source {i}:** *{doc_name}*")
                        
#                         content = source['content']
#                         if len(content) > 500:
#                             content = content[:500] + "..."
#                         st.code(content)
            
#             # Show an indication if result came from web search
#             if is_web_search:
#                 st.info("ℹ️ This answer was generated from web search results because I couldn't find enough information in your documents.")

# # Get RAG instance
# rag = initialize_rag_system()

# # Sidebar for document upload and management
# with st.sidebar:
#     st.subheader("📂 Document Management")
    
#     # File uploader
#     uploaded_files = st.file_uploader(
#         "Upload documents", 
#         type=["pdf", "txt", "docx", "png", "jpg", "jpeg", "bmp", "tiff"], 
#         accept_multiple_files=True,
#         key="file_uploader"
#     )
    
#     # Process button
#     if st.button("Process Documents", key="process_btn"):
#         if uploaded_files:
#             with st.spinner("Processing documents..."):
#                 try:
#                     saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
#                     # Display summaries immediately
#                     if summaries:
#                         st.subheader("📝 Quick Summaries")
#                         for item in summaries:
#                             icon = "🖼️" if item["type"] == "image" else "📄"
#                             with st.expander(f"{icon} {item['filename']}", expanded=False):
#                                 st.write(item["summary"])
                    
#                     new_documents = rag.document_processor.process_documents(saved_paths)
#                     st.session_state.all_documents.extend(new_documents)
#                     rag.create_vector_store(st.session_state.all_documents)
#                     rag.create_rag_chain(chain_type="stuff", k=4)
#                     st.session_state.documents_processed = True
#                     st.session_state.processed_files.extend(processed_files)
#                     st.success(f"✅ Processed {len(uploaded_files)} files!")
#                 except Exception as e:
#                     st.error(f"Error processing documents: {e}")
#         else:
#             st.warning("Please upload files first")
    
#     # Web search mode selection
#     st.subheader("🌐 Web Search Settings")
#     web_search_mode = st.radio(
#         "When to use web search:",
#         options=["Auto (when documents don't have answers)", "Always", "Never"],
#         index=0,
#         key="web_search_radio"
#     )
    
#     # Map radio selection to state
#     if web_search_mode == "Auto (when documents don't have answers)":
#         st.session_state.web_search_mode = "auto"
#     elif web_search_mode == "Always":
#         st.session_state.web_search_mode = "always"
#     else:
#         st.session_state.web_search_mode = "never"
    
#     # Display processed documents
#     if st.session_state.get("documents_processed", False):
#         st.subheader("📚 Processed Documents")
#         standard_docs = []
#         image_ocr_docs = []
#         image_desc_docs = []
        
#         for doc in st.session_state.get("processed_files", []):
#             if doc.get("type") == "image_ocr":
#                 image_ocr_docs.append(doc)
#             elif doc.get("type") == "image_description":
#                 image_desc_docs.append(doc)
#             else:
#                 standard_docs.append(doc)
        
#         if standard_docs:
#             st.write("**Standard Documents:**")
#             for doc in standard_docs:
#                 st.write(f"📄 {doc['original_name']}")
        
#         if image_ocr_docs:
#             st.write("**Images with Text (OCR):**")
#             for doc in image_ocr_docs:
#                 st.write(f"🖼️ {doc['original_name']}")
        
#         if image_desc_docs:
#             st.write("**Images with Descriptions:**")
#             for doc in image_desc_docs:
#                 st.write(f"🎨 {doc['original_name']}")
        
#         st.write(f"**Total documents:** {len(standard_docs) + len(image_ocr_docs) + len(image_desc_docs)}")
    
#     # Clear chat history button
#     if st.button("Clear Chat History"):
#         st.session_state.chat_history = []
#         st.rerun()

# # Main chat interface
# st.subheader("💬 Chat with Your Documents + Web")

# # Display chat history
# for chat in st.session_state.chat_history[-10:]:  # Show last 10 messages
#     display_chat_message(
#         chat["role"], 
#         chat["content"], 
#         chat.get("sources"),
#         chat.get("from_web", False)
#     )

# # Chat input
# user_question = st.chat_input("Ask a question about your documents or anything else...")

# if user_question:
#     # Add user question to chat history and display
#     st.session_state.chat_history.append({"role": "user", "content": user_question})
#     display_chat_message("user", user_question)
    
#     # Determine if we should force web search
#     force_web_search = st.session_state.web_search_mode == "always"
#     skip_web_search = st.session_state.web_search_mode == "never"
    
#     # Get and display assistant response
#     with st.spinner("Thinking..."):
#         try:
#             start_time = time.time()
            
#             if skip_web_search:
#                 # Only use RAG, even if it has low confidence
#                 result = rag._execute_rag_query(user_question)
#                 if "error" in result:
#                     # If RAG fails, show a helpful message
#                     result = {
#                         "answer": "I couldn't find an answer in your documents. You might want to try enabling web search to expand my knowledge.",
#                         "sources": [],
#                         "query_time": time.time() - start_time,
#                         "from_web": False
#                     }
#             else:
#                 # Use the enhanced query function with auto web search fallback
#                 result = rag.query(user_question, force_web_search=force_web_search)
            
#             query_time = round(time.time() - start_time, 2)
            
#             # Add assistant response to chat history
#             chat_entry = {
#                 "role": "assistant",
#                 "content": result['answer'],
#                 "sources": result.get('sources', []),
#                 "from_web": result.get('from_web', False),
#                 "query_time": query_time
#             }
#             st.session_state.chat_history.append(chat_entry)
            
#             # Display the response
#             display_chat_message(
#                 "assistant", 
#                 result['answer'], 
#                 result.get('sources', []),
#                 result.get('from_web', False)
#             )
            
#             # Show response time with appropriate icon
#             if result.get('from_web', False):
#                 st.success(f"⏱️ Response time: {query_time} seconds (🌐 Web Search)")
#             else:
#                 st.success(f"⏱️ Response time: {query_time} seconds (📚 Documents)")
                
#         except Exception as e:
#             st.error(f"Error processing query: {e}")


# import streamlit as st
# import os
# import tempfile
# from werkzeug.utils import secure_filename
# import time
# from PIL import Image
# import fitz  # PyMuPDF for PDF handling
# import easyocr
# import ssl
# from io import BytesIO
# from typing import List, Dict, Any

# # Import enhanced RAG system
# from rag_system.enhanced_rag_system import EnhancedRAGSystem

# # SSL configuration for EasyOCR
# ssl._create_default_https_context = ssl._create_unverified_context

# # Streamlit page config
# st.set_page_config(page_title="RAG-SWAT with Web Search", layout="wide")
# st.title("📘 RAG-SWAT with Web Search")

# # Constants
# UPLOAD_FOLDER = tempfile.mkdtemp()
# ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
# NEGATIVE_ANSWER_PHRASES = [
#     "no information", "not contain", "does not mention", 
#     "sorry", "i cannot", "don't know", "no context", "unable to"
# ]

# # Initialize components
# @st.cache_resource
# def initialize_rag_system():
#     api_key = "AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"  # Replace with your API key
#     return EnhancedRAGSystem(
#         api_key=api_key,
#         model_name="gemini-1.5-flash",
#         embedding_model="all-MiniLM-L6-v2",
#         confidence_threshold=0.7
#     )

# @st.cache_resource
# def get_ocr_reader():
#     return easyocr.Reader(['en'])

# # Initialize session state
# if 'all_documents' not in st.session_state:
#     st.session_state.all_documents = []
# if 'document_names' not in st.session_state:
#     st.session_state.document_names = []
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = []
# if 'chat_history' not in st.session_state:
#     st.session_state.chat_history = []
# if 'web_search_mode' not in st.session_state:
#     st.session_state.web_search_mode = "auto"

# # Utility functions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_image(image_bytes):
#     """Extract text from image using EasyOCR"""
#     try:
#         reader = get_ocr_reader()
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
#             temp.write(image_bytes)
#             temp_path = temp.name
        
#         result = reader.readtext(temp_path)
#         text = "\n".join([entry[1] for entry in result])
#         os.unlink(temp_path)
#         return text
#     except Exception as e:
#         st.error(f"OCR Error: {e}")
#         return ""

# def generate_short_summary(text, title, is_image=False):
#     """Generate a very short summary using the LLM"""
#     if not text or len(text.strip()) < 10:
#         return "Not enough content to generate summary"
    
#     try:
#         rag = initialize_rag_system()
#         prompt = f"""
#         Provide a concise summary (max 100 words) of this {'image' if is_image else 'document'} titled '{title}':
#         Focus on key points and main subject:
        
#         {text[:5000]}
#         """
#         response = rag.llm.invoke(prompt)
#         summary = response.content if hasattr(response, "content") else response
#         return ' '.join(summary.split()[:100])
#     except Exception as e:
#         st.error(f"Summary error: {e}")
#         return "Summary unavailable"

# def process_pdf(file_path, filename):
#     """Process PDF with OCR fallback"""
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         for page in doc:
#             text += page.get_text() or ""
#         doc.close()
#         return text if len(text) > 100 else process_pdf_with_ocr(file_path)
#     except Exception as e:
#         st.error(f"PDF Error: {e}")
#         return process_pdf_with_ocr(file_path)

# def process_pdf_with_ocr(file_path):
#     """Process PDF using OCR"""
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         reader = get_ocr_reader()
#         for page_num in range(len(doc)):
#             pix = page.get_pixmap()
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
#                 temp.write(pix.tobytes("png"))
#                 temp_path = temp.name
#             result = reader.readtext(temp_path)
#             text += "\n".join([entry[1] for entry in result])
#             os.unlink(temp_path)
#         doc.close()
#         return text
#     except Exception as e:
#         st.error(f"OCR PDF Error: {e}")
#         return ""

# def process_files(uploaded_files):
#     saved_paths = []
#     processed_files = []
#     summaries = []
    
#     rag = initialize_rag_system()
    
#     for uploaded_file in uploaded_files:
#         if not uploaded_file or not allowed_file(uploaded_file.name):
#             continue
            
#         filename = secure_filename(uploaded_file.name)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
        
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())
        
#         file_ext = filename.rsplit('.', 1)[1].lower()
#         is_image = file_ext in {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        
#         try:
#             if is_image:
#                 # Process images
#                 extracted_text = extract_text_from_image(uploaded_file.getvalue())
#                 if extracted_text:
#                     processed_text = f"IMAGE TEXT:\n{extracted_text}"
#                     summary = generate_short_summary(extracted_text, filename, True)
#                 else:
#                     processed_text = f"IMAGE DESCRIPTION:\nCould not extract text"
#                     summary = "Image with no detectable text"
                
#                 text_filename = f"{filename}_processed.txt"
#                 text_path = os.path.join(UPLOAD_FOLDER, text_filename)
#                 with open(text_path, 'w', encoding='utf-8') as f:
#                     f.write(processed_text)
                
#                 saved_paths.append(text_path)
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": text_filename,
#                     "type": "image"
#                 })
            
#             elif file_ext == 'pdf':
#                 # Process PDFs
#                 text = process_pdf(file_path, filename)
#                 if text:
#                     saved_paths.append(file_path)
#                     summary = generate_short_summary(text, filename)
#                     processed_files.append({
#                         "original_name": filename,
#                         "processed_name": filename,
#                         "type": "pdf"
#                     })
#                 else:
#                     st.warning(f"Could not process PDF: {filename}")
#                     continue
            
#             else:
#                 # Process other documents
#                 saved_paths.append(file_path)
#                 if file_ext == 'txt':
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         text = f.read(5000)
#                         summary = generate_short_summary(text, filename)
#                 else:
#                     summary = f"Uploaded {file_ext.upper()} file"
                
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": filename,
#                     "type": "document"
#                 })
            
#             if summary:
#                 summaries.append({
#                     "filename": filename,
#                     "summary": summary,
#                     "type": "image" if is_image else "document"
#                 })
                
#         except Exception as e:
#             st.error(f"Error processing {filename}: {e}")
    
#     return saved_paths, processed_files, summaries

# def display_chat_message(role, content, sources=None, from_web=False):
#     """Display a chat message with sources"""
#     if role == "user":
#         with st.chat_message("user", avatar="🧑"):
#             st.markdown(content)
#     else:
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(content)
            
#             if sources:
#                 with st.expander("Sources", expanded=False):
#                     for i, source in enumerate(sources, 1):
#                         doc_name = source['metadata'].get('document_name', 'Unknown')
#                         icon = "🌐" if "Web:" in doc_name else "📄"
#                         url = source['metadata'].get('source_url', '')
                        
#                         if url:
#                             st.markdown(f"**{icon} [{doc_name}]({url})**")
#                         else:
#                             st.markdown(f"**{icon} {doc_name}**")
                        
#                         content = source['content']
#                         st.code(content[:500] + ("..." if len(content) > 500 else ""))
            
#             if from_web:
#                 st.info("ℹ️ Web search results used (not in your documents)")

# # Main application
# rag = initialize_rag_system()

# # Sidebar
# with st.sidebar:
#     st.subheader("📂 Document Management")
    
#     uploaded_files = st.file_uploader(
#         "Upload documents",
#         type=list(ALLOWED_EXTENSIONS),
#         accept_multiple_files=True
#     )
    
#     if st.button("Process Documents"):
#         if uploaded_files:
#             with st.spinner("Processing..."):
#                 try:
#                     saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
#                     if summaries:
#                         st.subheader("📝 Summaries")
#                         for item in summaries:
#                             icon = "🖼️" if item["type"] == "image" else "📄"
#                             with st.expander(f"{icon} {item['filename']}"):
#                                 st.write(item["summary"])
                    
#                     new_docs = rag.document_processor.process_documents(saved_paths)
#                     st.session_state.all_documents.extend(new_docs)
#                     rag.create_vector_store(st.session_state.all_documents)
#                     rag.create_rag_chain()
#                     st.session_state.processed_files.extend(processed_files)
#                     st.session_state.documents_processed = True
#                     st.success(f"✅ Processed {len(uploaded_files)} files!")
#                 except Exception as e:
#                     st.error(f"Processing error: {e}")
#         else:
#             st.warning("Please upload files first")
    
#     st.subheader("🌐 Web Search")
#     web_mode = st.radio(
#         "Web Search Mode",
#         options=["Auto (when needed)", "Always", "Never"],
#         index=0
#     )
#     st.session_state.web_search_mode = web_mode.split()[0].lower()
    
#     if st.session_state.processed_files:
#         st.subheader("📚 Processed Files")
#         for file in st.session_state.processed_files:
#             icon = "🖼️" if file["type"] == "image" else "📄"
#             st.write(f"{icon} {file['original_name']}")

# # Main chat interface
# st.subheader("💬 Chat with Your Documents")

# # Display chat history
# for chat in st.session_state.chat_history[-10:]:
#     display_chat_message(
#         chat["role"],
#         chat["content"],
#         chat.get("sources"),
#         chat.get("from_web", False)
#     )

# # Chat input
# user_question = st.chat_input("Ask a question...")
# force_web = st.session_state.web_search_mode == "always"
# disable_web = st.session_state.web_search_mode == "never"

# if user_question:
#     st.session_state.chat_history.append({"role": "user", "content": user_question})
#     display_chat_message("user", user_question)
    
#     with st.spinner("Thinking..."):
#         try:
#             start_time = time.time()
            
#             if disable_web:
#                 result = rag._execute_rag_query(user_question)
#                 if not result["sources"]:
#                     result["answer"] = "No information found in documents. Try enabling web search."
#             else:
#                 result = rag.query(user_question, force_web_search=force_web)
            
#             # Store and display response
#             st.session_state.chat_history.append({
#                 "role": "assistant",
#                 "content": result["answer"],
#                 "sources": result.get("sources", []),
#                 "from_web": result.get("from_web", False)
#             })
            
#             display_chat_message(
#                 "assistant",
#                 result["answer"],
#                 result.get("sources", []),
#                 result.get("from_web", False)
#             )
            
#             # Show response time
#             response_time = time.time() - start_time
#             source = "🌐 Web" if result.get("from_web") else "📚 Documents"
#             st.success(f"⏱️ {response_time:.1f}s | {source}")
            
#         except Exception as e:
#             st.error(f"Error: {e}")



# import streamlit as st
# import os
# import tempfile
# from werkzeug.utils import secure_filename
# import time
# from PIL import Image
# import fitz  # PyMuPDF for PDF handling
# import easyocr
# import ssl
# from io import BytesIO
# from typing import List, Dict, Any

# # Import enhanced RAG system
# from rag_system.enhanced_rag_system import EnhancedRAGSystem

# # SSL configuration for EasyOCR
# ssl._create_default_https_context = ssl._create_unverified_context

# # Streamlit page config
# st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
# st.title("🤖 Conversational RAG Assistant")

# # Constants
# UPLOAD_FOLDER = tempfile.mkdtemp()
# ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# # Initialize components
# @st.cache_resource
# def initialize_rag_system():
#     api_key = "AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"  # Replace with your API key
#     return EnhancedRAGSystem(
#         api_key=api_key,
#         model_name="gemini-1.5-flash",
#         embedding_model="all-MiniLM-L6-v2",
#         confidence_threshold=0.7
#     )

# @st.cache_resource
# def get_ocr_reader():
#     return easyocr.Reader(['en'])

# # Initialize session state
# if 'conversation' not in st.session_state:
#     st.session_state.conversation = []
# if 'awaiting_clarification' not in st.session_state:
#     st.session_state.awaiting_clarification = False
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = []
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'web_search_mode' not in st.session_state:
#     st.session_state.web_search_mode = "auto"

# # Utility functions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_image(image_bytes):
#     """Extract text from image using EasyOCR"""
#     try:
#         reader = get_ocr_reader()
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
#             temp.write(image_bytes)
#             temp_path = temp.name
        
#         result = reader.readtext(temp_path)
#         text = "\n".join([entry[1] for entry in result])
#         os.unlink(temp_path)
#         return text
#     except Exception as e:
#         st.error(f"OCR Error: {e}")
#         return ""

# def generate_short_summary(text, title, is_image=False):
#     """Generate a very short summary using the LLM"""
#     if not text or len(text.strip()) < 10:
#         return "Not enough content to generate summary"
    
#     try:
#         rag = initialize_rag_system()
#         prompt = f"""
#         Provide a concise summary (max 100 words) of this {'image' if is_image else 'document'} titled '{title}':
#         Focus on key points and main subject:
        
#         {text[:5000]}
#         """
#         response = rag.llm.invoke(prompt)
#         summary = response.content if hasattr(response, "content") else response
#         return ' '.join(summary.split()[:100])
#     except Exception as e:
#         st.error(f"Summary error: {e}")
#         return "Summary unavailable"

# def process_pdf(file_path, filename):
#     """Process PDF with OCR fallback"""
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         for page in doc:
#             text += page.get_text() or ""
#         doc.close()
#         return text if len(text) > 100 else process_pdf_with_ocr(file_path)
#     except Exception as e:
#         st.error(f"PDF Error: {e}")
#         return process_pdf_with_ocr(file_path)

# def process_pdf_with_ocr(file_path):
#     """Process PDF using OCR"""
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         reader = get_ocr_reader()
#         for page_num in range(len(doc)):
#             pix = page.get_pixmap()
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
#                 temp.write(pix.tobytes("png"))
#                 temp_path = temp.name
#             result = reader.readtext(temp_path)
#             text += "\n".join([entry[1] for entry in result])
#             os.unlink(temp_path)
#         doc.close()
#         return text
#     except Exception as e:
#         st.error(f"OCR PDF Error: {e}")
#         return ""

# def process_files(uploaded_files):
#     saved_paths = []
#     processed_files = []
#     summaries = []
    
#     rag = initialize_rag_system()
    
#     for uploaded_file in uploaded_files:
#         if not uploaded_file or not allowed_file(uploaded_file.name):
#             continue
            
#         filename = secure_filename(uploaded_file.name)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
        
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())
        
#         file_ext = filename.rsplit('.', 1)[1].lower()
#         is_image = file_ext in {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        
#         try:
#             if is_image:
#                 # Process images
#                 extracted_text = extract_text_from_image(uploaded_file.getvalue())
#                 if extracted_text:
#                     processed_text = f"IMAGE TEXT:\n{extracted_text}"
#                     summary = generate_short_summary(extracted_text, filename, True)
#                 else:
#                     processed_text = f"IMAGE DESCRIPTION:\nCould not extract text"
#                     summary = "Image with no detectable text"
                
#                 text_filename = f"{filename}_processed.txt"
#                 text_path = os.path.join(UPLOAD_FOLDER, text_filename)
#                 with open(text_path, 'w', encoding='utf-8') as f:
#                     f.write(processed_text)
                
#                 saved_paths.append(text_path)
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": text_filename,
#                     "type": "image"
#                 })
            
#             elif file_ext == 'pdf':
#                 # Process PDFs
#                 text = process_pdf(file_path, filename)
#                 if text:
#                     saved_paths.append(file_path)
#                     summary = generate_short_summary(text, filename)
#                     processed_files.append({
#                         "original_name": filename,
#                         "processed_name": filename,
#                         "type": "pdf"
#                     })
#                 else:
#                     st.warning(f"Could not process PDF: {filename}")
#                     continue
            
#             else:
#                 # Process other documents
#                 saved_paths.append(file_path)
#                 if file_ext == 'txt':
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         text = f.read(5000)
#                         summary = generate_short_summary(text, filename)
#                 else:
#                     summary = f"Uploaded {file_ext.upper()} file"
                
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": filename,
#                     "type": "document"
#                 })
            
#             if summary:
#                 summaries.append({
#                     "filename": filename,
#                     "summary": summary,
#                     "type": "image" if is_image else "document"
#                 })
                
#         except Exception as e:
#             st.error(f"Error processing {filename}: {e}")
    
#     return saved_paths, processed_files, summaries

# # def generate_clarification_question(query: str) -> str:
# #     """Generate a context-aware follow-up question using LLM"""
# #     rag = initialize_rag_system()
# #     prompt = f"""
# #     The user asked: "{query}"
    
# #     Generate one concise, natural follow-up question that would help:
# #     1. Understand their specific needs
# #     2. Identify the most relevant aspect
# #     3. Provide a tailored response
    
# #     Examples:
# #     - "Are you looking for technical details or practical applications?"
# #     - "Should I focus on current information or historical background?"
# #     - "Would beginner-level or advanced concepts be more helpful?"
    
# #     Your question (just the question, no other text):
# #     """
# #     response = rag.llm.invoke(prompt)
# #     return response.content.strip()

# # def needs_clarification(conversation: List[Dict[str, str]]) -> bool:
# #     """Determine if clarification would improve response quality"""
# #     if len(conversation) < 2:
# #         return True
    
# #     last_exchange = "\n".join([msg["content"] for msg in conversation[-2:]])
# #     rag = initialize_rag_system()
# #     prompt = f"""
# #     Analyze this conversation snippet. Should the assistant ask for 
# #     clarification before answering? Consider:
# #     - Is the query broad or ambiguous?
# #     - Would context help provide a better answer?
    
# #     Respond only 'yes' or 'no'.
    
# #     Conversation:
# #     {last_exchange}
    
# #     Decision: """
    
# #     response = rag.llm.invoke(prompt)
# #     return response.content.strip().lower() == 'yes'

# def generate_clarification_question(query: str, conversation: List[Dict[str, str]]) -> str:
#     """Generate a context-aware follow-up question using LLM with limit tracking"""
#     # Count how many times we've asked for clarification in this conversation thread
#     clarification_count = sum(1 for msg in conversation if msg.get("action") == "clarify")
    
#     # If we've already asked twice, return None to indicate we should answer directly
#     if clarification_count >= 2:
#         return None
    
#     rag = initialize_rag_system()
#     prompt = f"""
#     The user asked: "{query}"
    
#     Generate exactly one concise follow-up question that would help provide a better answer.
#     Focus on identifying the most specific aspect they're interested in.
    
#     Examples:
#     - "Are you looking for technical details or practical applications?"
#     - "Should I focus on current information or historical context?"
#     - "Would you like beginner-level or advanced information?"
    
#     Your question (just the question, no other text):
#     """
#     response = rag.llm.invoke(prompt)
#     return response.content.strip()

# def needs_clarification(conversation: List[Dict[str, str]]) -> bool:
#     """More selective about when to ask for clarification"""
#     if len(conversation) < 2:
#         return True
    
#     # Count existing clarifications in this thread
#     clarification_count = sum(1 for msg in conversation if msg.get("action") == "clarify")
#     if clarification_count >= 2:
#         return False
    
#     last_query = conversation[-1]["content"].lower()
    
#     # Only ask for clarification on broad question starters
#     broad_phrases = ["what is", "tell me about", "explain", "who is", "how to"]
#     if not any(phrase in last_query for phrase in broad_phrases):
#         return False
    
#     # Check if the query is actually broad
#     prompt = f"""
#     Is this query too broad to answer directly without clarification? 
#     Respond only 'yes' or 'no'.
#     Query: "{last_query}"
#     """
    
#     rag = initialize_rag_system()
#     response = rag.llm.invoke(prompt)
#     return response.content.strip().lower() == 'yes'

# # [Rest of your existing code remains the same until the user input handling]


# def display_chat_message(role, content, sources=None, from_web=False):
#     """Display a chat message with sources"""
#     if role == "user":
#         with st.chat_message("user", avatar="🧑"):
#             st.markdown(content)
#     else:
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(content)
            
#             if sources:
#                 with st.expander("Sources", expanded=False):
#                     for i, source in enumerate(sources, 1):
#                         doc_name = source['metadata'].get('document_name', 'Unknown')
#                         icon = "🌐" if "Web:" in doc_name else "📄"
#                         url = source['metadata'].get('source_url', '')
                        
#                         if url:
#                             st.markdown(f"**{icon} [{doc_name}]({url})**")
#                         else:
#                             st.markdown(f"**{icon} {doc_name}**")
                        
#                         content = source['content']
#                         st.code(content[:500] + ("..." if len(content) > 500 else ""))
            
#             if from_web:
#                 st.info("ℹ️ Web search results used (not in your documents)")

# # Main application
# rag = initialize_rag_system()

# # Sidebar
# with st.sidebar:
#     st.subheader("📂 Document Management")
    
#     uploaded_files = st.file_uploader(
#         "Upload documents",
#         type=list(ALLOWED_EXTENSIONS),
#         accept_multiple_files=True
#     )
    
#     if st.button("Process Documents"):
#         if uploaded_files:
#             with st.spinner("Processing..."):
#                 try:
#                     saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
#                     if summaries:
#                         st.subheader("📝 Summaries")
#                         for item in summaries:
#                             icon = "🖼️" if item["type"] == "image" else "📄"
#                             with st.expander(f"{icon} {item['filename']}"):
#                                 st.write(item["summary"])
                    
#                     new_docs = rag.document_processor.process_documents(saved_paths)
#                     rag.create_vector_store(new_docs)
#                     rag.create_rag_chain()
#                     st.session_state.processed_files.extend(processed_files)
#                     st.session_state.documents_processed = True
#                     st.success(f"✅ Processed {len(uploaded_files)} files!")
#                 except Exception as e:
#                     st.error(f"Processing error: {e}")
#         else:
#             st.warning("Please upload files first")
    
#     st.subheader("🌐 Web Search")
#     web_mode = st.radio(
#         "Web Search Mode",
#         options=["Auto (when needed)", "Always", "Never"],
#         index=0
#     )
#     st.session_state.web_search_mode = web_mode.split()[0].lower()
    
#     if st.session_state.processed_files:
#         st.subheader("📚 Processed Files")
#         for file in st.session_state.processed_files:
#             icon = "🖼️" if file["type"] == "image" else "📄"
#             st.write(f"{icon} {file['original_name']}")

# # Main chat interface
# st.subheader("💬 Chat with Your Documents")

# # Display chat history
# for msg in st.session_state.conversation:
#     display_chat_message(
#         msg["role"],
#         msg["content"],
#         msg.get("sources"),
#         msg.get("from_web", False)
#     )

# # User input
# user_input = st.chat_input("Ask me anything...")
# force_web = st.session_state.web_search_mode == "always"
# disable_web = st.session_state.web_search_mode == "never"

# # if user_input:
# #     # Add user message to conversation
# #     st.session_state.conversation.append({"role": "user", "content": user_input})
# #     display_chat_message("user", user_input)
    
# #     with st.spinner("Thinking..."):
# #         try:
# #             start_time = time.time()
            
# #             # First determine if we need clarification
# #             if needs_clarification(st.session_state.conversation):
# #                 clarification = generate_clarification_question(user_input)
# #                 response = {
# #                     "answer": clarification,
# #                     "action": "clarify",
# #                     "sources": [],
# #                     "from_web": False
# #                 }
# #                 st.session_state.awaiting_clarification = True
# #             else:
# #                 # Get full answer
# #                 if disable_web:
# #                     response = rag._execute_rag_query(user_input)
# #                     if not response["sources"]:
# #                         response["answer"] = "No information found in documents. Try enabling web search."
# #                 else:
# #                     response = rag.query(user_input, force_web_search=force_web)
# #                 st.session_state.awaiting_clarification = False
            
# #             # Add response to conversation
# #             st.session_state.conversation.append({
# #                 "role": "assistant",
# #                 "content": response["answer"],
# #                 "sources": response.get("sources", []),
# #                 "from_web": response.get("from_web", False)
# #             })
            
# #             # Display response
# #             display_chat_message(
# #                 "assistant",
# #                 response["answer"],
# #                 response.get("sources", []),
# #                 response.get("from_web", False)
# #             )
            
# #             # Show response time
# #             response_time = time.time() - start_time
# #             source = "🌐 Web" if response.get("from_web") else "📚 Documents"
# #             st.success(f"⏱️ {response_time:.1f}s | {source}")
            
# #         except Exception as e:
# #             st.error(f"Error: {e}")


# if user_input:
#     # Add user message to conversation
#     st.session_state.conversation.append({"role": "user", "content": user_input})
#     display_chat_message("user", user_input)
    
#     with st.spinner("Thinking..."):
#         try:
#             start_time = time.time()
            
#             # First determine if we need clarification
#             if needs_clarification(st.session_state.conversation):
#                 clarification = generate_clarification_question(user_input, st.session_state.conversation)
                
#                 if clarification:  # Only ask if we haven't reached our limit
#                     response = {
#                         "answer": clarification,
#                         "action": "clarify",
#                         "sources": [],
#                         "from_web": False
#                     }
#                     st.session_state.awaiting_clarification = True
#                 else:
#                     # If we've reached our clarification limit, answer directly
#                     if disable_web:
#                         response = rag._execute_rag_query(user_input)
#                         if not response["sources"]:
#                             response["answer"] = "No information found in documents. Try enabling web search."
#                     else:
#                         response = rag.query(user_input, force_web_search=force_web)
#                     st.session_state.awaiting_clarification = False
#             else:
#                 # Get full answer directly
#                 if disable_web:
#                     response = rag._execute_rag_query(user_input)
#                     if not response["sources"]:
#                         response["answer"] = "No information found in documents. Try enabling web search."
#                 else:
#                     response = rag.query(user_input, force_web_search=force_web)
#                 st.session_state.awaiting_clarification = False
            
#             # Add response to conversation
#             st.session_state.conversation.append({
#                 "role": "assistant",
#                 "content": response["answer"],
#                 "sources": response.get("sources", []),
#                 "from_web": response.get("from_web", False)
#             })
            
#             # Display response
#             display_chat_message(
#                 "assistant",
#                 response["answer"],
#                 response.get("sources", []),
#                 response.get("from_web", False)
#             )
            
#             # Show response time
#             response_time = time.time() - start_time
#             source = "🌐 Web" if response.get("from_web") else "📚 Documents"
#             st.success(f"⏱️ {response_time:.1f}s | {source}")
            
#         except Exception as e:
#             st.error(f"Error: {e}")


# # Status indicators
# if st.session_state.conversation:
#     last_msg = st.session_state.conversation[-1]
#     if last_msg["role"] == "assistant":
#         if st.session_state.awaiting_clarification:
#             st.caption("🔍 Waiting for more details to provide the best answer...")
#         elif last_msg.get("from_web"):
#             st.caption("🌐 Answer supplemented with web search")
#         else:
#             st.caption("📚 Answer from your documents")



# import streamlit as st
# import os
# import tempfile
# from werkzeug.utils import secure_filename
# import time
# from PIL import Image
# import fitz  # PyMuPDF for PDF handling
# import easyocr
# import ssl
# from io import BytesIO
# from typing import List, Dict, Any

# # Import enhanced RAG system
# from rag_system.enhanced_rag_system import EnhancedRAGSystem

# # SSL configuration for EasyOCR
# ssl._create_default_https_context = ssl._create_unverified_context

# # Streamlit page config
# st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
# st.title("🤖 Conversational RAG Assistant")

# # Constants
# UPLOAD_FOLDER = tempfile.mkdtemp()
# ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# # Initialize components
# @st.cache_resource
# def initialize_rag_system():
#     api_key = "AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"  # Replace with your API key
#     return EnhancedRAGSystem(
#         api_key=api_key,
#         model_name="gemini-1.5-flash",
#         embedding_model="all-MiniLM-L6-v2",
#         confidence_threshold=0.7
#     )

# @st.cache_resource
# def get_ocr_reader():
#     return easyocr.Reader(['en'])

# # Initialize session state
# if 'conversation' not in st.session_state:
#     st.session_state.conversation = []
# if 'awaiting_clarification' not in st.session_state:
#     st.session_state.awaiting_clarification = False
# if 'processed_files' not in st.session_state:
#     st.session_state.processed_files = []
# if 'documents_processed' not in st.session_state:
#     st.session_state.documents_processed = False
# if 'web_search_mode' not in st.session_state:
#     st.session_state.web_search_mode = "auto"
# if 'conversation_context' not in st.session_state:
#     st.session_state.conversation_context = {}
# if 'last_query_topic' not in st.session_state:
#     st.session_state.last_query_topic = None

# # Utility functions
# def allowed_file(filename):
#     return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_image(image_bytes):
#     """Extract text from image using EasyOCR"""
#     try:
#         reader = get_ocr_reader()
#         with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
#             temp.write(image_bytes)
#             temp_path = temp.name
        
#         result = reader.readtext(temp_path)
#         text = "\n".join([entry[1] for entry in result])
#         os.unlink(temp_path)
#         return text
#     except Exception as e:
#         st.error(f"OCR Error: {e}")
#         return ""

# def generate_short_summary(text, title, is_image=False):
#     """Generate a very short summary using the LLM"""
#     if not text or len(text.strip()) < 10:
#         return "Not enough content to generate summary"
    
#     try:
#         rag = initialize_rag_system()
#         prompt = f"""
#         Provide a concise summary (max 100 words) of this {'image' if is_image else 'document'} titled '{title}':
#         Focus on key points and main subject:
        
#         {text[:5000]}
#         """
#         response = rag.llm.invoke(prompt)
#         summary = response.content if hasattr(response, "content") else response
#         return ' '.join(summary.split()[:100])
#     except Exception as e:
#         st.error(f"Summary error: {e}")
#         return "Summary unavailable"

# def process_pdf(file_path, filename):
#     """Process PDF with OCR fallback"""
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         for page in doc:
#             text += page.get_text() or ""
#         doc.close()
#         return text if len(text) > 100 else process_pdf_with_ocr(file_path)
#     except Exception as e:
#         st.error(f"PDF Error: {e}")
#         return process_pdf_with_ocr(file_path)

# def process_pdf_with_ocr(file_path):
#     """Process PDF using OCR"""
#     try:
#         doc = fitz.open(file_path)
#         text = ""
#         reader = get_ocr_reader()
#         for page_num in range(len(doc)):
#             page = doc[page_num]
#             pix = page.get_pixmap()
#             with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
#                 temp.write(pix.tobytes("png"))
#                 temp_path = temp.name
#             result = reader.readtext(temp_path)
#             text += "\n".join([entry[1] for entry in result])
#             os.unlink(temp_path)
#         doc.close()
#         return text
#     except Exception as e:
#         st.error(f"OCR PDF Error: {e}")
#         return ""

# def process_files(uploaded_files):
#     saved_paths = []
#     processed_files = []
#     summaries = []
    
#     rag = initialize_rag_system()
    
#     for uploaded_file in uploaded_files:
#         if not uploaded_file or not allowed_file(uploaded_file.name):
#             continue
            
#         filename = secure_filename(uploaded_file.name)
#         file_path = os.path.join(UPLOAD_FOLDER, filename)
        
#         with open(file_path, "wb") as f:
#             f.write(uploaded_file.getvalue())
        
#         file_ext = filename.rsplit('.', 1)[1].lower()
#         is_image = file_ext in {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        
#         try:
#             if is_image:
#                 # Process images
#                 extracted_text = extract_text_from_image(uploaded_file.getvalue())
#                 if extracted_text:
#                     processed_text = f"IMAGE TEXT:\n{extracted_text}"
#                     summary = generate_short_summary(extracted_text, filename, True)
#                 else:
#                     processed_text = f"IMAGE DESCRIPTION:\nCould not extract text"
#                     summary = "Image with no detectable text"
                
#                 text_filename = f"{filename}_processed.txt"
#                 text_path = os.path.join(UPLOAD_FOLDER, text_filename)
#                 with open(text_path, 'w', encoding='utf-8') as f:
#                     f.write(processed_text)
                
#                 saved_paths.append(text_path)
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": text_filename,
#                     "type": "image"
#                 })
            
#             elif file_ext == 'pdf':
#                 # Process PDFs
#                 text = process_pdf(file_path, filename)
#                 if text:
#                     saved_paths.append(file_path)
#                     summary = generate_short_summary(text, filename)
#                     processed_files.append({
#                         "original_name": filename,
#                         "processed_name": filename,
#                         "type": "pdf"
#                     })
#                 else:
#                     st.warning(f"Could not process PDF: {filename}")
#                     continue
            
#             else:
#                 # Process other documents
#                 saved_paths.append(file_path)
#                 if file_ext == 'txt':
#                     with open(file_path, 'r', encoding='utf-8') as f:
#                         text = f.read(5000)
#                         summary = generate_short_summary(text, filename)
#                 else:
#                     summary = f"Uploaded {file_ext.upper()} file"
                
#                 processed_files.append({
#                     "original_name": filename,
#                     "processed_name": filename,
#                     "type": "document"
#                 })
            
#             if summary:
#                 summaries.append({
#                     "filename": filename,
#                     "summary": summary,
#                     "type": "image" if is_image else "document"
#                 })
                
#         except Exception as e:
#             st.error(f"Error processing {filename}: {e}")
    
#     return saved_paths, processed_files, summaries

# def extract_main_topic(query):
#     """Extract the main topic from a query"""
#     rag = initialize_rag_system()
#     prompt = f"""
#     Extract the main topic or subject from this query. Return only the topic itself,
#     no other words or explanation.
    
#     Query: {query}
    
#     Main topic:
#     """
#     response = rag.llm.invoke(prompt)
#     return response.content.strip()

# def maintain_conversation_context(query, conversation_history):
#     """Update the conversation context with the current query"""
#     # Extract the main topic
#     topic = extract_main_topic(query)
    
#     # Store it for future reference
#     if topic:
#         st.session_state.last_query_topic = topic
#         if topic not in st.session_state.conversation_context:
#             st.session_state.conversation_context[topic] = []
        
#         # Add this query to the topic's context
#         st.session_state.conversation_context[topic].append(query)

# def resolve_query_with_context(query, conversation_history):
#     """Resolve the query using conversation context"""
#     # If we don't have enough history, just return the original query
#     if len(conversation_history) < 2:
#         return query
    
#     # Check if this is a follow-up question (short query that might need context)
#     is_potential_followup = len(query.split()) <= 5 and ('?' in query or query.lower().startswith('what') or 
#                                                           query.lower().startswith('who') or 
#                                                           query.lower().startswith('how') or
#                                                           query.lower().startswith('why') or
#                                                           query.lower().startswith('when') or
#                                                           query.lower().startswith('where'))
    
#     if not is_potential_followup:
#         return query
    
#     # Use the LLM to resolve the query with context
#     rag = initialize_rag_system()
    
#     # Get the last few messages to provide context
#     recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
#     context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages])
    
#     prompt = f"""
#     Given the conversation context below and the most recent query, determine if the query needs additional context to be properly understood.
#     If it does, rewrite the query with the necessary context. If not, return the original query unchanged.
    
#     Conversation:
#     {context}
    
#     Latest query: "{query}"
    
#     Resolved query (include all context needed for a standalone question):
#     """
    
#     response = rag.llm.invoke(prompt)
#     resolved_query = response.content.strip()
    
#     # Only use the resolved query if it's substantially different and longer
#     if resolved_query != query and len(resolved_query) > len(query):
#         return resolved_query
    
#     # If the topic is known and this is a short follow-up, explicitly include the topic
#     if st.session_state.last_query_topic and is_potential_followup:
#         return f"{query} about {st.session_state.last_query_topic}"
    
#     # Otherwise return the original
#     return query

# def generate_clarification_question(query: str, conversation: List[Dict[str, str]]) -> str:
#     """Generate a context-aware follow-up question using LLM with limit tracking"""
#     # Count how many times we've asked for clarification in this conversation thread
#     clarification_count = sum(1 for msg in conversation if msg.get("action") == "clarify")
    
#     # If we've already asked twice, return None to indicate we should answer directly
#     if clarification_count >= 2:
#         return None
    
#     # Store the main topic for context
#     st.session_state.last_query_topic = extract_main_topic(query)
    
#     rag = initialize_rag_system()
#     prompt = f"""
#     The user asked: "{query}"
    
#     Generate exactly one concise follow-up question that would help provide a better answer.
#     Focus on identifying the most specific aspect they're interested in.
    
#     Examples:
#     - "Are you looking for technical details or practical applications?"
#     - "Should I focus on current information or historical context?"
#     - "Would you like beginner-level or advanced information?"
    
#     Your question (just the question, no other text):
#     """
#     response = rag.llm.invoke(prompt)
#     return response.content.strip()

# def needs_clarification(conversation: List[Dict[str, str]]) -> bool:
#     """More selective about when to ask for clarification"""
#     if len(conversation) < 1:
#         return False
    
#     # Count existing clarifications in this thread
#     clarification_count = sum(1 for msg in conversation if msg.get("action") == "clarify")
#     if clarification_count >= 2:
#         return False
    
#     # Check if this is a follow-up to a clarification request
#     if st.session_state.awaiting_clarification and len(conversation) >= 2:
#         return False
    
#     last_query = conversation[-1]["content"].lower()
    
#     # Skip clarification for very short queries or questions about clarification itself
#     if len(last_query.split()) <= 3 or "what do you mean" in last_query:
#         return False
    
#     # Only ask for clarification on broad question starters
#     broad_phrases = ["what is", "tell me about", "explain", "who is", "how to"]
#     if not any(phrase in last_query for phrase in broad_phrases):
#         return False
    
#     # Check if the query is actually broad
#     prompt = f"""
#     Is this query too broad to answer directly without clarification? 
#     Consider if there are multiple possible interpretations or aspects that the user might be interested in.
#     Respond only 'yes' or 'no'.
#     Query: "{last_query}"
#     """
    
#     rag = initialize_rag_system()
#     response = rag.llm.invoke(prompt)
#     return response.content.strip().lower() == 'yes'

# def display_chat_message(role, content, sources=None, from_web=False):
#     """Display a chat message with sources"""
#     if role == "user":
#         with st.chat_message("user", avatar="🧑"):
#             st.markdown(content)
#     else:
#         with st.chat_message("assistant", avatar="🤖"):
#             st.markdown(content)
            
#             if sources:
#                 with st.expander("Sources", expanded=False):
#                     for i, source in enumerate(sources, 1):
#                         doc_name = source['metadata'].get('document_name', 'Unknown')
#                         icon = "🌐" if "Web:" in doc_name else "📄"
#                         url = source['metadata'].get('source_url', '')
                        
#                         if url:
#                             st.markdown(f"**{icon} [{doc_name}]({url})**")
#                         else:
#                             st.markdown(f"**{icon} {doc_name}**")
                        
#                         content = source['content']
#                         st.code(content[:500] + ("..." if len(content) > 500 else ""))
            
#             if from_web:
#                 st.info("ℹ️ Web search results used (not in your documents)")

# # Main application
# rag = initialize_rag_system()

# # Sidebar
# with st.sidebar:
#     st.subheader("📂 Document Management")
    
#     uploaded_files = st.file_uploader(
#         "Upload documents",
#         type=list(ALLOWED_EXTENSIONS),
#         accept_multiple_files=True
#     )
    
#     if st.button("Process Documents"):
#         if uploaded_files:
#             with st.spinner("Processing..."):
#                 try:
#                     saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
#                     if summaries:
#                         st.subheader("📝 Summaries")
#                         for item in summaries:
#                             icon = "🖼️" if item["type"] == "image" else "📄"
#                             with st.expander(f"{icon} {item['filename']}"):
#                                 st.write(item["summary"])
                    
#                     new_docs = rag.document_processor.process_documents(saved_paths)
#                     rag.create_vector_store(new_docs)
#                     rag.create_rag_chain()
#                     st.session_state.processed_files.extend(processed_files)
#                     st.session_state.documents_processed = True
#                     st.success(f"✅ Processed {len(uploaded_files)} files!")
#                 except Exception as e:
#                     st.error(f"Processing error: {e}")
#         else:
#             st.warning("Please upload files first")
    
#     st.subheader("🌐 Web Search")
#     web_mode = st.radio(
#         "Web Search Mode",
#         options=["Auto (when needed)", "Always", "Never"],
#         index=0
#     )
#     st.session_state.web_search_mode = web_mode.split()[0].lower()
    
#     if st.session_state.processed_files:
#         st.subheader("📚 Processed Files")
#         for file in st.session_state.processed_files:
#             icon = "🖼️" if file["type"] == "image" else "📄"
#             st.write(f"{icon} {file['original_name']}")

# # Main chat interface
# st.subheader("💬 Chat with Your Documents")

# # Display chat history
# for msg in st.session_state.conversation:
#     display_chat_message(
#         msg["role"],
#         msg["content"],
#         msg.get("sources"),
#         msg.get("from_web", False)
#     )

# # User input
# user_input = st.chat_input("Ask me anything...")
# force_web = st.session_state.web_search_mode == "always"
# disable_web = st.session_state.web_search_mode == "never"

# if user_input:
#     # Add user message to conversation
#     st.session_state.conversation.append({"role": "user", "content": user_input})
#     display_chat_message("user", user_input)
    
#     with st.spinner("Thinking..."):
#         try:
#             start_time = time.time()
            
#             # Update conversation context
#             maintain_conversation_context(user_input, st.session_state.conversation)
            
#             # Check if we're responding to a clarification
#             if st.session_state.awaiting_clarification:
#                 st.session_state.awaiting_clarification = False
                
#                 # Use context from the conversation to resolve the query
#                 resolved_query = resolve_query_with_context(user_input, st.session_state.conversation)
                
#                 # Get full answer with the resolved query
#                 if disable_web:
#                     response = rag._execute_rag_query(resolved_query)
#                     if not response["sources"]:
#                         response["answer"] = "No information found in documents. Try enabling web search."
#                 else:
#                     response = rag.query(resolved_query, force_web_search=force_web)
#             else:
#                 # First determine if we need clarification
#                 if needs_clarification(st.session_state.conversation):
#                     clarification = generate_clarification_question(user_input, st.session_state.conversation)
                    
#                     if clarification:  # Only ask if we haven't reached our limit
#                         response = {
#                             "answer": clarification,
#                             "action": "clarify",
#                             "sources": [],
#                             "from_web": False
#                         }
#                         st.session_state.awaiting_clarification = True
#                     else:
#                         # If we've reached our clarification limit, answer directly
#                         resolved_query = resolve_query_with_context(user_input, st.session_state.conversation)
                        
#                         if disable_web:
#                             response = rag._execute_rag_query(resolved_query)
#                             if not response["sources"]:
#                                 response["answer"] = "No information found in documents. Try enabling web search."
#                         else:
#                             response = rag.query(resolved_query, force_web_search=force_web)
#                         st.session_state.awaiting_clarification = False
#                 else:
#                     # Resolve the query using context from the conversation
#                     resolved_query = resolve_query_with_context(user_input, st.session_state.conversation)
                    
#                     # Get full answer directly
#                     if disable_web:
#                         response = rag._execute_rag_query(resolved_query)
#                         if not response["sources"]:
#                             response["answer"] = "No information found in documents. Try enabling web search."
#                     else:
#                         response = rag.query(resolved_query, force_web_search=force_web)
#                     st.session_state.awaiting_clarification = False
            
#             # Add response to conversation with action flag if clarifying
#             st.session_state.conversation.append({
#                 "role": "assistant",
#                 "content": response["answer"],
#                 "sources": response.get("sources", []),
#                 "from_web": response.get("from_web", False),
#                 "action": response.get("action", "answer")
#             })
            
#             # Display response
#             display_chat_message(
#                 "assistant",
#                 response["answer"],
#                 response.get("sources", []),
#                 response.get("from_web", False)
#             )
            
#             # Show response time
#             response_time = time.time() - start_time
#             source = "🌐 Web" if response.get("from_web") else "📚 Documents"
#             st.success(f"⏱️ {response_time:.1f}s | {source}")
            
#         except Exception as e:
#             st.error(f"Error: {e}")

# # Status indicators
# if st.session_state.conversation:
#     last_msg = st.session_state.conversation[-1]
#     if last_msg["role"] == "assistant":
#         if st.session_state.awaiting_clarification:
#             st.caption("🔍 Waiting for more details to provide the best answer...")
#         elif last_msg.get("from_web"):
#             st.caption("🌐 Answer supplemented with web search")
#         else:
#             st.caption("📚 Answer from your documents")



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
from typing import List, Dict, Any
from transformers import pipeline

# Import enhanced RAG system
from rag_system.enhanced_rag_system import EnhancedRAGSystem

# SSL configuration for EasyOCR
ssl._create_default_https_context = ssl._create_unverified_context

# Streamlit page config
st.set_page_config(page_title="Conversational RAG Assistant", layout="wide")
st.title("🤖 Conversational RAG Assistant")

# Constants
UPLOAD_FOLDER = tempfile.mkdtemp()
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize components
@st.cache_resource
def initialize_rag_system():
    api_key = "AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk"  # Replace with your API key
    return EnhancedRAGSystem(
        api_key=api_key,
        model_name="gemini-1.5-flash",
        embedding_model="all-MiniLM-L6-v2",
        confidence_threshold=0.7
    )

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
if 'conversation' not in st.session_state:
    st.session_state.conversation = []
if 'awaiting_clarification' not in st.session_state:
    st.session_state.awaiting_clarification = False
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []
if 'documents_processed' not in st.session_state:
    st.session_state.documents_processed = False
if 'web_search_mode' not in st.session_state:
    st.session_state.web_search_mode = "auto"
if 'conversation_context' not in st.session_state:
    st.session_state.conversation_context = {}
if 'last_query_topic' not in st.session_state:
    st.session_state.last_query_topic = None

# Utility functions
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
    """Generate a very short summary using the LLM"""
    if not text or len(text.strip()) < 10:
        return "Not enough content to generate summary"
    
    try:
        rag = initialize_rag_system()
        if is_image:
            prompt = f"""
            Provide a concise summary (max 100 words) of this image titled '{title}' 
            based on its visual description or extracted text:
            
            {text[:5000]}
            """
        else:
            prompt = f"""
            Provide a concise summary (max 100 words) of this document titled '{title}':
            Focus on key points and main subject:
            
            {text[:5000]}
            """
        response = rag.llm.invoke(prompt)
        summary = response.content if hasattr(response, "content") else response
        return ' '.join(summary.split()[:100])
    except Exception as e:
        st.error(f"Summary error: {e}")
        return "Summary unavailable"

def process_pdf(file_path, filename):
    """Process PDF with OCR fallback"""
    try:
        doc = fitz.open(file_path)
        text = ""
        needs_ocr = False
        
        # First try regular text extraction
        for page in doc:
            page_text = page.get_text()
            if page_text.strip():
                text += page_text
            else:
                needs_ocr = True
                break
        
        doc.close()
        
        # If we detected pages needing OCR or got very little text
        if needs_ocr or len(text.strip()) < 100:
            return process_pdf_with_ocr(file_path), True
        
        return text, False
    except Exception as e:
        st.error(f"PDF Error: {e}")
        return process_pdf_with_ocr(file_path), True

def process_pdf_with_ocr(file_path):
    """Process PDF using OCR"""
    try:
        doc = fitz.open(file_path)
        text = ""
        reader = get_ocr_reader()
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap()
            with tempfile.NamedTemporaryFile(delete=False, suffix='.png') as temp:
                temp.write(pix.tobytes("png"))
                temp_path = temp.name
            result = reader.readtext(temp_path)
            text += f"Page {page_num+1}:\n" + "\n".join([entry[1] for entry in result]) + "\n\n"
            os.unlink(temp_path)
        doc.close()
        return text
    except Exception as e:
        st.error(f"OCR PDF Error: {e}")
        return ""

def process_files(uploaded_files):
    saved_paths = []
    processed_files = []
    summaries = []
    
    rag = initialize_rag_system()
    
    for uploaded_file in uploaded_files:
        if not uploaded_file or not allowed_file(uploaded_file.name):
            continue
            
        filename = secure_filename(uploaded_file.name)
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getvalue())
        
        file_ext = filename.rsplit('.', 1)[1].lower()
        is_image = file_ext in {'png', 'jpg', 'jpeg', 'bmp', 'tiff'}
        
        try:
            if is_image:
                # Process images with OCR and VLM
                extracted_text = extract_text_from_image(uploaded_file.getvalue())
                
                if extracted_text and len(extracted_text.strip()) > 10:
                    # Process with OCR if text was found
                    processed_text = f"IMAGE TEXT:\n{extracted_text}"
                    summary = generate_short_summary(extracted_text, filename, True)
                
                    text_filename = f"{filename}_ocr.txt"
                    text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                    with open(text_path, 'w', encoding='utf-8') as f:
                        f.write(processed_text)
                    
                    saved_paths.append(text_path)
                    processed_files.append({
                        "original_name": filename,
                        "processed_name": text_filename,
                        "type": "image_ocr"
                    })
                else:
                    # If no text found, use VLM to describe the image
                    description = describe_image(uploaded_file.getvalue())
                    
                    if description:
                        processed_text = f"IMAGE DESCRIPTION:\n{description}"
                        summary = generate_short_summary(description, filename, True)
                    
                        text_filename = f"{filename}_description.txt"
                        text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(processed_text)
                        
                        saved_paths.append(text_path)
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": text_filename,
                            "type": "image_description"
                        })
                    else:
                        st.warning(f"Could not process image: {filename}")
                        continue
            
            elif file_ext == 'pdf':
                # Process PDFs
                text, used_ocr = process_pdf(file_path, filename)
                if text:
                    if used_ocr:
                        # Save OCR-processed text to a separate file
                        text_filename = f"{filename}_ocr.txt"
                        text_path = os.path.join(UPLOAD_FOLDER, text_filename)
                        with open(text_path, 'w', encoding='utf-8') as f:
                            f.write(text)
                        
                        saved_paths.append(text_path)
                        summary = generate_short_summary(text, filename)
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": text_filename,
                            "type": "pdf_ocr"
                        })
                    else:
                        saved_paths.append(file_path)
                        summary = generate_short_summary(text, filename)
                        processed_files.append({
                            "original_name": filename,
                            "processed_name": filename,
                            "type": "document"
                        })
                else:
                    st.warning(f"Could not process PDF: {filename}")
                    continue
            
            else:
                # Process other documents
                saved_paths.append(file_path)
                if file_ext == 'txt':
                    with open(file_path, 'r', encoding='utf-8') as f:
                        text = f.read(5000)
                        summary = generate_short_summary(text, filename)
                else:
                    summary = f"Uploaded {file_ext.upper()} file"
                
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
            
            if summary:
                summaries.append({
                    "filename": filename,
                    "summary": summary,
                    "type": "image" if is_image else "document"
                })
                
        except Exception as e:
            st.error(f"Error processing {filename}: {e}")
    
    return saved_paths, processed_files, summaries

def extract_main_topic(query):
    """Extract the main topic from a query"""
    rag = initialize_rag_system()
    prompt = f"""
    Extract the main topic or subject from this query. Return only the topic itself,
    no other words or explanation.
    
    Query: {query}
    
    Main topic:
    """
    response = rag.llm.invoke(prompt)
    return response.content.strip()

def maintain_conversation_context(query, conversation_history):
    """Update the conversation context with the current query"""
    # Extract the main topic
    topic = extract_main_topic(query)
    
    # Store it for future reference
    if topic:
        st.session_state.last_query_topic = topic
        if topic not in st.session_state.conversation_context:
            st.session_state.conversation_context[topic] = []
        
        # Add this query to the topic's context
        st.session_state.conversation_context[topic].append(query)

def resolve_query_with_context(query, conversation_history):
    """Resolve the query using conversation context"""
    # If we don't have enough history, just return the original query
    if len(conversation_history) < 2:
        return query
    
    # Check if this is a follow-up question (short query that might need context)
    is_potential_followup = len(query.split()) <= 5 and ('?' in query or query.lower().startswith('what') or 
                                                          query.lower().startswith('who') or 
                                                          query.lower().startswith('how') or
                                                          query.lower().startswith('why') or
                                                          query.lower().startswith('when') or
                                                          query.lower().startswith('where'))
    
    if not is_potential_followup:
        return query
    
    # Use the LLM to resolve the query with context
    rag = initialize_rag_system()
    
    # Get the last few messages to provide context
    recent_messages = conversation_history[-4:] if len(conversation_history) >= 4 else conversation_history
    context = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in recent_messages])
    
    prompt = f"""
    Given the conversation context below and the most recent query, determine if the query needs additional context to be properly understood.
    If it does, rewrite the query with the necessary context. If not, return the original query unchanged.
    
    Conversation:
    {context}
    
    Latest query: "{query}"
    
    Resolved query (include all context needed for a standalone question):
    """
    
    response = rag.llm.invoke(prompt)
    resolved_query = response.content.strip()
    
    # Only use the resolved query if it's substantially different and longer
    if resolved_query != query and len(resolved_query) > len(query):
        return resolved_query
    
    # If the topic is known and this is a short follow-up, explicitly include the topic
    if st.session_state.last_query_topic and is_potential_followup:
        return f"{query} about {st.session_state.last_query_topic}"
    
    # Otherwise return the original
    return query

def generate_clarification_question(query: str, conversation: List[Dict[str, str]]) -> str:
    """Generate a context-aware follow-up question using LLM with limit tracking"""
    # Count how many times we've asked for clarification in this conversation thread
    clarification_count = sum(1 for msg in conversation if msg.get("action") == "clarify")
    
    # If we've already asked twice, return None to indicate we should answer directly
    if clarification_count >= 2:
        return None
    
    # Store the main topic for context
    st.session_state.last_query_topic = extract_main_topic(query)
    
    rag = initialize_rag_system()
    prompt = f"""
    The user asked: "{query}"
    
    Generate exactly one concise follow-up question that would help provide a better answer.
    Focus on identifying the most specific aspect they're interested in.
    
    Examples:
    - "Are you looking for technical details or practical applications?"
    - "Should I focus on current information or historical context?"
    - "Would you like beginner-level or advanced information?"
    
    Your question (just the question, no other text):
    """
    response = rag.llm.invoke(prompt)
    return response.content.strip()

def needs_clarification(conversation: List[Dict[str, str]]) -> bool:
    """More selective about when to ask for clarification"""
    if len(conversation) < 1:
        return False
    
    # Count existing clarifications in this thread
    clarification_count = sum(1 for msg in conversation if msg.get("action") == "clarify")
    if clarification_count >= 2:
        return False
    
    # Check if this is a follow-up to a clarification request
    if st.session_state.awaiting_clarification and len(conversation) >= 2:
        return False
    
    last_query = conversation[-1]["content"].lower()
    
    # Skip clarification for very short queries or questions about clarification itself
    if len(last_query.split()) <= 3 or "what do you mean" in last_query:
        return False
    
    # Only ask for clarification on broad question starters
    broad_phrases = ["what is", "tell me about", "explain", "who is", "how to"]
    if not any(phrase in last_query for phrase in broad_phrases):
        return False
    
    # Check if the query is actually broad
    prompt = f"""
    Is this query too broad to answer directly without clarification? 
    Consider if there are multiple possible interpretations or aspects that the user might be interested in.
    Respond only 'yes' or 'no'.
    Query: "{last_query}"
    """
    
    rag = initialize_rag_system()
    response = rag.llm.invoke(prompt)
    return response.content.strip().lower() == 'yes'

def display_chat_message(role, content, sources=None, from_web=False):
    """Display a chat message with sources"""
    if role == "user":
        with st.chat_message("user", avatar="🧑"):
            st.markdown(content)
    else:
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(content)
            
            if sources:
                with st.expander("Sources", expanded=False):
                    for i, source in enumerate(sources, 1):
                        doc_name = source['metadata'].get('document_name', 'Unknown')
                        
                        # Choose appropriate icon based on source
                        if "Web:" in doc_name:
                            icon = "🌐"
                        elif "(OCR)" in doc_name or "ocr" in doc_name.lower():
                            icon = "🖼️ (Text from Image)"
                        elif "(Image Description)" in doc_name or "description" in doc_name.lower():
                            icon = "🖼️ (Image Description)"
                        else:
                            icon = "📄"
                        
                        url = source['metadata'].get('source_url', '')
                        
                        if url:
                            st.markdown(f"**{icon} [{doc_name}]({url})**")
                        else:
                            st.markdown(f"**{icon} {doc_name}**")
                        
                        content = source['content']
                        st.code(content[:500] + ("..." if len(content) > 500 else ""))
            
            if from_web:
                st.info("ℹ️ Web search results used (not in your documents)")

# Main application
rag = initialize_rag_system()

# Sidebar
with st.sidebar:
    st.subheader("📂 Document Management")
    
    uploaded_files = st.file_uploader(
        "Upload documents",
        type=list(ALLOWED_EXTENSIONS),
        accept_multiple_files=True
    )
    
    if st.button("Process Documents"):
        if uploaded_files:
            with st.spinner("Processing..."):
                try:
                    saved_paths, processed_files, summaries = process_files(uploaded_files)
                    
                    if summaries:
                        st.subheader("📝 Summaries")
                        for item in summaries:
                            icon = "🖼️" if item["type"] == "image" else "📄"
                            with st.expander(f"{icon} {item['filename']}"):
                                st.write(item["summary"])
                    
                    new_docs = rag.document_processor.process_documents(saved_paths)
                    rag.create_vector_store(new_docs)
                    rag.create_rag_chain()
                    st.session_state.processed_files.extend(processed_files)
                    st.session_state.documents_processed = True
                    st.success(f"✅ Processed {len(uploaded_files)} files!")
                except Exception as e:
                    st.error(f"Processing error: {e}")
        else:
            st.warning("Please upload files first")
    
    st.subheader("🌐 Web Search")
    web_mode = st.radio(
        "Web Search Mode",
        options=["Auto (when needed)", "Always", "Never"],
        index=0
    )
    st.session_state.web_search_mode = web_mode.split()[0].lower()
    
    if st.session_state.processed_files:
        st.subheader("📚 Processed Files")
        
        # Categorize files by type
        standard_docs = []
        image_ocr_docs = []
        image_desc_docs = []
        
        for file in st.session_state.processed_files:
            if file["type"] == "image_ocr" or file["type"] == "pdf_ocr":
                image_ocr_docs.append(file)
            elif file["type"] == "image_description":
                image_desc_docs.append(file)
            else:
                standard_docs.append(file)
        
        # Display categorized files
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

# Main chat interface
st.subheader("💬 Chat with Your Documents")

# Display chat history
for msg in st.session_state.conversation:
    display_chat_message(
        msg["role"],
        msg["content"],
        msg.get("sources"),
        msg.get("from_web", False)
    )

# User input
user_input = st.chat_input("Ask me anything...")
force_web = st.session_state.web_search_mode == "always"
disable_web = st.session_state.web_search_mode == "never"

if user_input:
    # Add user message to conversation
    st.session_state.conversation.append({"role": "user", "content": user_input})
    display_chat_message("user", user_input)
    
    with st.spinner("Thinking..."):
        try:
            start_time = time.time()
            
            # Update conversation context
            maintain_conversation_context(user_input, st.session_state.conversation)
            
            # Check if we're responding to a clarification
            if st.session_state.awaiting_clarification:
                st.session_state.awaiting_clarification = False
                
                # Use context from the conversation to resolve the query
                resolved_query = resolve_query_with_context(user_input, st.session_state.conversation)
                
                # Get full answer with the resolved query
                if disable_web:
                    response = rag._execute_rag_query(resolved_query)
                    if not response["sources"]:
                        response["answer"] = "No information found in documents. Try enabling web search."
                else:
                    response = rag.query(resolved_query, force_web_search=force_web)
            else:
                # First determine if we need clarification
                if needs_clarification(st.session_state.conversation):
                    clarification = generate_clarification_question(user_input, st.session_state.conversation)
                    
                    if clarification:  # Only ask if we haven't reached our limit
                        response = {
                            "answer": clarification,
                            "action": "clarify",
                            "sources": [],
                            "from_web": False
                        }
                        st.session_state.awaiting_clarification = True
                    else:
                        # If we've reached our clarification limit, answer directly
                        resolved_query = resolve_query_with_context(user_input, st.session_state.conversation)
                        
                        if disable_web:
                            response = rag._execute_rag_query(resolved_query)
                            if not response["sources"]:
                                response["answer"] = "No information found in documents. Try enabling web search."
                        else:
                            response = rag.query(resolved_query, force_web_search=force_web)
                        st.session_state.awaiting_clarification = False
                else:
                    # Resolve the query using context from the conversation
                    resolved_query = resolve_query_with_context(user_input, st.session_state.conversation)
                    
                    # Get full answer directly
                    if disable_web:
                        response = rag._execute_rag_query(resolved_query)
                        if not response["sources"]:
                            response["answer"] = "No information found in documents. Try enabling web search."
                    else:
                        response = rag.query(resolved_query, force_web_search=force_web)
                    st.session_state.awaiting_clarification = False
            
            # Add response to conversation with action flag if clarifying
            st.session_state.conversation.append({
                "role": "assistant",
                "content": response["answer"],
                "sources": response.get("sources", []),
                "from_web": response.get("from_web", False),
                "action": response.get("action", "answer")
            })
            
            # Display response
            display_chat_message(
                "assistant",
                response["answer"],
                response.get("sources", []),
                response.get("from_web", False)
            )
            
            # Show response time
            response_time = time.time() - start_time
            source = "🌐 Web" if response.get("from_web") else "📚 Documents"
            st.success(f"⏱️ {response_time:.1f}s | {source}")
            
        except Exception as e:
            st.error(f"Error: {e}")

# Status indicators
if st.session_state.conversation:
    last_msg = st.session_state.conversation[-1]
    if last_msg["role"] == "assistant":
        if st.session_state.awaiting_clarification:
            st.caption("🔍 Waiting for more details to provide the best answer...")
        elif last_msg.get("from_web"):
            st.caption("🌐 Answer supplemented with web search")
        else:
            st.caption("📚 Answer from your documents")