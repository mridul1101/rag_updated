import os
import time
from typing import List, Dict, Any,Optional
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from config.config import EMBEDDING_MODEL, INDEX_PATH, MODEL_NAME,logger
from document_processing.document_processor import DocumentProcessor
from retrieval.retriever import DateFilteredRetriever

class RAGSystem:
    def __init__(self, api_key: str, model_name: str = MODEL_NAME, 
                 embedding_model: str = EMBEDDING_MODEL,
                 index_path: str = INDEX_PATH):
        os.environ["GOOGLE_API_KEY"] = api_key
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index_path = index_path
        self.vector_store = None
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        self.document_processor = DocumentProcessor()
        logger.info(f"RAG System initialized with model {model_name} and embedding model {embedding_model}")

    def create_vector_store(self, documents: List[Document]) -> None:
        """Create a vector store from Documents."""
        start_time = time.time()
        logger.info(f"Creating vector store from {len(documents)} documents")

        try:
            texts = [doc.page_content for doc in documents]
            metadatas = [doc.metadata for doc in documents]
            self.vector_store = FAISS.from_texts(texts, self.embeddings, metadatas=metadatas)
            logger.info(f"Vector store created in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            logger.error(f"Error creating vector store: {str(e)}")

    def create_rag_chain(self, chain_type: str = "stuff", k: int = 4) -> None:
        """Create a RAG chain for question answering."""
        if not self.vector_store:
            logger.error("No vector store available. Load or create one first.")
            return

        logger.info(f"Creating RAG chain with {chain_type} chain type")
        retriever = DateFilteredRetriever(vector_store=self.vector_store, k=k)
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type=chain_type,
            retriever=retriever,
            return_source_documents=True
        )
        logger.info("RAG chain created successfully")


    def query(self, query_text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Query the RAG system and return the response with source documents."""
        if not hasattr(self, 'rag_chain'):
            logger.error("RAG chain not initialized. Call create_rag_chain first.")
            return {"error": "RAG chain not initialized"}
        
        logger.info(f"Querying RAG system with: {query_text}")
        start_time = time.time()
        
        try:
            # Add conversation history to the query context
            if conversation_history:
                context = "\n".join([f"User: {entry['user']}\nSystem: {entry['system']}" for entry in conversation_history])
                query_with_context = f"{context}\nUser: {query_text}"
            else:
                query_with_context = query_text
            
            # Execute query
            response = self.rag_chain.invoke({"query": query_with_context})
            
            # Format source information for return
            sources = []
            if "source_documents" in response:
                for doc in response["source_documents"]:
                    sources.append({
                        "content": doc.page_content[:150] + "..." if len(doc.page_content) > 150 else doc.page_content,
                        "metadata": doc.metadata
                    })
            
            result = {
                "answer": response.get("result", "No result returned"),
                "sources": sources,
                "query_time": time.time() - start_time
            }
            
            # Update conversation history
            self.conversation_history.append({"user": query_text, "system": result["answer"]})
            
            logger.info(f"Query completed in {result['query_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error during query: {str(e)}")
            return {"error": str(e), "query_time": time.time() - start_time}

