from flask import Flask, request, jsonify
from rag_system.rag_system import RAGSystem
from config.config import API_KEY, MODEL_NAME, EMBEDDING_MODEL, INDEX_PATH
from flask_cors import CORS
from langchain.globals import set_debug
from werkzeug.utils import secure_filename
import os
import tempfile
import time
import pytesseract
from PIL import Image
import google.generativeai as genai
import fitz  # PyMuPDF for PDF handling
import easyocr
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

# Then initialize EasyOCR
reader = easyocr.Reader(['en'])
# Set debug mode
set_debug(True)

app = Flask(__name__)
CORS(app)

# Create temp folder for uploaded files
UPLOAD_FOLDER = tempfile.mkdtemp()
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Configure allowed file extensions
ALLOWED_EXTENSIONS = {'pdf', 'txt', 'docx', 'png', 'jpg', 'jpeg', 'bmp', 'tiff'}

# Initialize the RAG system
rag = RAGSystem(
    api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk",
    model_name="gemini-2.0-flash",
    embedding_model="all-MiniLM-L6-v2"
)

# Initialize Gemini for image processing and summarization
genai.configure(api_key="AIzaSyDd02LbjboeF8AeRX46oTW8Z9gc1Yx6YCk")
model = genai.GenerativeModel('gemini-2.0-flash')

# Track all processed documents
all_documents = []
document_names = []

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# def extract_text_from_image(image_path):
#     """Extract text from image using OCR"""
#     try:
#         image = Image.open(image_path)
#         text = pytesseract.image_to_string(image)
#         return text
#     except Exception as e:
#         print(f"OCR Error: {e}")
#         return ""
def extract_text_from_image(image_path):
    """Extract text from image using EasyOCR"""
    try:
        reader = easyocr.Reader(['en'])  # Initialize for English
        result = reader.readtext(image_path)
        text = "\n".join([entry[1] for entry in result])
        return text
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""

def summarize_text(text, title):
    """Generate a summary of the extracted text using Gemini"""
    if not text or len(text.strip()) < 100:  # Skip summarization for very short texts
        return text
    
    try:
        prompt = f"""
        Please summarize the following text extracted from the document titled '{title}'. 
        Maintain key facts, figures, and important information:
        
        {text[:10000]}  # Limiting to first 10000 chars to avoid token limits
        """
        
        response = model.generate_content(prompt)
        summary = response.text
        
        # Combine summary with original text to maintain searchability
        result = f"SUMMARY: {summary}\n\nORIGINAL TEXT: {text[:5000]}"  # Include partial original text
        return result
    except Exception as e:
        print(f"Summarization Error: {e}")
        return text

@app.route("/upload", methods=["POST"])
def upload_files():
    """Handle file uploads from the Streamlit app"""
    global all_documents, document_names
    
    if 'files' not in request.files:
        return jsonify({"error": "No files part in the request"}), 400
        
    files = request.files.getlist('files')
    
    if not files or files[0].filename == '':
        return jsonify({"error": "No files selected"}), 400
    
    # Save uploaded files temporarily and process them
    saved_paths = []
    processed_files = []
    
    for file in files:
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            file_ext = filename.rsplit('.', 1)[1].lower()
            
            # Process images with OCR and summarization
            if file_ext in ['png', 'jpg', 'jpeg', 'bmp', 'tiff']:
                extracted_text = extract_text_from_image(file_path)
                if extracted_text:
                    # Create a text file with the extracted content
                    text_filename = f"{filename}_ocr.txt"
                    text_path = os.path.join(app.config['UPLOAD_FOLDER'], text_filename)
                    
                    # Summarize extracted text
                    processed_text = summarize_text(extracted_text, filename)
                    
                    with open(text_path, 'w', encoding='utf-8') as text_file:
                        text_file.write(processed_text)
                    
                    saved_paths.append(text_path)
                    document_names.append(f"{filename} (OCR)")
                    processed_files.append({
                        "original_name": filename,
                        "processed_name": text_filename,
                        "type": "image_ocr"
                    })
            else:
                # Standard document processing
                saved_paths.append(file_path)
                document_names.append(filename)
                processed_files.append({
                    "original_name": filename,
                    "processed_name": filename,
                    "type": "document"
                })
    
    # Process the documents with the RAG system
    new_documents = rag.document_processor.process_documents(saved_paths)
    
    # Add to all documents
    all_documents.extend(new_documents)
    
    # Create or update vector store
    rag.create_vector_store(all_documents)
    
    # Create RAG chain
    rag.create_rag_chain(chain_type="stuff", k=4)
    
    return jsonify({
        "status": "success", 
        "document_names": document_names,
        "processed_files": processed_files
    })

@app.route("/query", methods=["POST"])
def query():
    # Get the user's question from the request
    data = request.json
    user_input = data.get("question")
    
    if not user_input:
        return jsonify({"error": "No question provided"}), 400
    
    # Query the RAG system
    result = rag.query(user_input)
    
    # Prepare the response
    response = {
        "answer": result["answer"],
        "sources": [
            {
                "content": source["content"],
                "document_name": source["metadata"].get("document_name", "Unknown"),
                "date": source["metadata"].get("date", "Unknown")
            }
            for source in result["sources"]
        ],
        "query_time": result["query_time"]
    }
    
    return jsonify(response)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)