# import os
# import re
# import time
# from typing import List, Dict, Any
# from langchain.schema import Document
# from langchain_community.document_loaders import PyPDFLoader
# from config.config import logger
# class DocumentProcessor:
#     def __init__(self):
#         pass

#     def clean_text(self, text: str) -> str:
#         """Clean and normalize the text."""
#         text = re.sub(r'\s+', ' ', text).strip()
#         text = re.sub(r'[^\w\s.,;!?]', '', text)
#         return text

#     def extract_date_from_filename(self, filename: str) -> str:
#         """Extract date from filename (e.g., 'HDFC_MF_Factsheet_April_2024.pdf')."""
#         parts = filename.split("_")
#         if len(parts) >= 4:
#             month = parts[-2]  # April
#             year = parts[-1].split(".")[0]  # 2024
#             return f"{month} {year}"
#         return "Unknown Date"

#     def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> Document:
#         """Process a single PDF document and return a Document with metadata."""
#         start_time = time.time()
#         logger.info(f"Processing document: {file_path}")

#         if metadata is None:
#             metadata = {"source": file_path}

#         try:
#             loader = PyPDFLoader(file_path)
#             pages = loader.load_and_split()
#             full_text = " ".join([page.page_content for page in pages])
#             full_text = self.clean_text(full_text)
#             date = self.extract_date_from_filename(os.path.basename(file_path))

#             metadata.update({
#                 "document_name": os.path.basename(file_path),
#                 "document_type": os.path.basename(os.path.dirname(file_path)),
#                 "date": date
#             })

#             document = Document(page_content=full_text, metadata=metadata)
#             logger.info(f"Processed document {file_path} in {time.time() - start_time:.2f} seconds")
#             return document

#         except Exception as e:
#             logger.error(f"Error processing {file_path}: {str(e)}")
#             return None

#     def process_documents(self, document_paths: List[str]) -> List[Document]:
#         """Process multiple documents and return all Documents."""
#         logger.info(f"Processing {len(document_paths)} documents")
#         all_documents = []

#         for doc_path in document_paths:
#             document = self.process_document(doc_path)
#             if document:
#                 all_documents.append(document)

#         logger.info(f"Processed {len(document_paths)} documents, created {len(all_documents)} total Documents")
#         return all_documents



import os
import re
import time
from typing import List, Dict, Any
from langchain.schema import Document
from langchain.text_splitter import MarkdownHeaderTextSplitter
from config.config import logger
from markitdown import MarkItDown

class DocumentProcessor:
    def __init__(self):
        # Initialize MarkItDown
        self.markitdown = MarkItDown(enable_plugins=False)

    def clean_text(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    def extract_date_from_filename(self, filename: str) -> str:
        parts = filename.split("_")
        if len(parts) >= 4:
            month = parts[-2]
            year = parts[-1].split(".")[0]
            return f"{month} {year}"
        return "Unknown Date"

    def chunk_markdown(self, markdown_text: str) -> List[Document]:
        # Split based on Markdown headers
        splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "Header1"), ("##", "Header2")])
        docs = splitter.split_text(markdown_text)
        return docs

    def process_document(self, file_path: str, metadata: Dict[str, Any] = None) -> List[Document]:
        start_time = time.time()
        logger.info(f"Processing document: {file_path}")

        if metadata is None:
            metadata = {"source": file_path}

        try:
            # Convert using MarkItDown
            result = self.markitdown.convert(file_path)
            markdown_text = result.markdown  # full Markdown content
            chunks = self.chunk_markdown(markdown_text)

            date = self.extract_date_from_filename(os.path.basename(file_path))
            enriched_chunks = []

            for chunk in chunks:
                enriched_chunks.append(Document(
                    page_content=self.clean_text(chunk.page_content),
                    metadata={
                        **metadata,
                        "document_name": os.path.basename(file_path),
                        "document_type": os.path.basename(os.path.dirname(file_path)),
                        "date": date
                    }
                ))

            logger.info(f"Processed {file_path} into {len(enriched_chunks)} chunks in {time.time() - start_time:.2f} seconds")
            return enriched_chunks

        except Exception as e:
            logger.error(f"Error processing {file_path}: {str(e)}")
            return []

    def process_documents(self, document_paths: List[str]) -> List[Document]:
        logger.info(f"Processing {len(document_paths)} documents")
        all_documents = []

        for doc_path in document_paths:
            documents = self.process_document(doc_path)
            all_documents.extend(documents)

        logger.info(f"Final document count after chunking: {len(all_documents)}")
        return all_documents
