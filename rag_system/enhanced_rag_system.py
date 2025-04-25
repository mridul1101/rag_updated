import os
import time
from typing import List, Dict, Any, Optional, Tuple
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
import numpy as np
from config.config import EMBEDDING_MODEL, INDEX_PATH, MODEL_NAME, logger
from document_processing.document_processor import DocumentProcessor
from retrieval.retriever import DateFilteredRetriever
from web_search.web_search_agent import WebSearchAgent

class EnhancedRAGSystem:
    def __init__(self, api_key: str, model_name: str = MODEL_NAME, 
                 embedding_model: str = EMBEDDING_MODEL,
                 index_path: str = INDEX_PATH,
                 confidence_threshold: float = 0.2):
        """Initialize the enhanced RAG system with web search capabilities."""
        os.environ["GOOGLE_API_KEY"] = api_key
        self.api_key = api_key
        self.confidence_threshold = confidence_threshold
        
        # Initialize LLM for RAG
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Initialize embeddings and document processor
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model)
        self.index_path = index_path
        self.vector_store = None
        self.document_processor = DocumentProcessor()
        
        # Initialize web search agent
        self.web_search_agent = WebSearchAgent(api_key=api_key, model_name=model_name)
        
        # Initialize conversation history
        self.conversation_history: List[Dict[str, str]] = []
        
        logger.info(f"Enhanced RAG System initialized with model {model_name} and embedding model {embedding_model}")

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

    def _evaluate_relevance(self, query: str, sources: List[Dict[str, Any]]) -> float:
        """
        Evaluate the relevance of RAG results to determine if web search is needed.
        Returns a confidence score between 0 and 1.
        """
        if not sources:
            return 0.0
        
        try:
            # Simple heuristic: Count how many sources have substantial content
            substantial_sources = sum(1 for source in sources if len(source.get('content', '')) > 50)
            source_ratio = substantial_sources / len(sources) if sources else 0
            
            # TODO: Could implement more sophisticated relevance checking with embeddings
            
            logger.info(f"Relevance evaluation: {source_ratio:.2f} ({substantial_sources}/{len(sources)} substantial sources)")
            return min(source_ratio, 1.0)  # Cap at 1.0
        except Exception as e:
            logger.error(f"Error evaluating relevance: {str(e)}")
            return 0.0

    
    def _execute_rag_query(self, query_text: str, conversation_history: Optional[List[Dict[str, str]]] = None) -> Dict[str, Any]:
        """Execute the RAG query and return the results."""
        if not hasattr(self, 'rag_chain'):
            logger.error("RAG chain not initialized. Call create_rag_chain first.")
            return {"error": "RAG chain not initialized"}
        
        logger.info(f"Executing RAG query: {query_text}")
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
                "query_time": time.time() - start_time,
                "from_web": False
            }
            
            logger.info(f"RAG query completed in {result['query_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error during RAG query: {str(e)}")
            return {"error": str(e), "query_time": time.time() - start_time, "from_web": False}
    
    def query(self, query_text: str, conversation_history: Optional[List[Dict[str, str]]] = None, 
              force_web_search: bool = False) -> Dict[str, Any]:
        """
        Query the enhanced RAG system with fallback to web search.
        
        Args:
            query_text: The user's question
            conversation_history: Optional conversation history
            force_web_search: If True, skip RAG and go directly to web search
            
        Returns:
            Dict containing answer, sources, and metadata
        """
        logger.info(f"Processing query with enhanced RAG system: {query_text}")
        start_time = time.time()
        
        try:
            # Step 1: Try RAG first (unless forced to use web search)
            if not force_web_search and hasattr(self, 'rag_chain'):
                rag_result = self._execute_rag_query(query_text, conversation_history)
                
                # Check if RAG result is satisfactory
                relevance_score = self._evaluate_relevance(query_text, rag_result.get('sources', []))
                logger.info(f"RAG relevance score: {relevance_score:.2f} (threshold: {self.confidence_threshold})")
                
                # If RAG is confident enough, return its result
                if relevance_score >= self.confidence_threshold:
                    logger.info("Using RAG result (confidence above threshold)")
                    
                    # Update conversation history
                    self.conversation_history.append({
                        "user": query_text, 
                        "system": rag_result["answer"]
                    })
                    
                    return rag_result
            
            # Step 2: If RAG failed or has low confidence, try web search
            logger.info("RAG result insufficient, falling back to web search")
            web_result = self.web_search_agent.query(query_text)
            
            # Add provenance information
            if not force_web_search:
                web_result["answer"] = "I couldn't find enough information in your documents, so I searched the web:\n\n" + web_result["answer"]
            
            # Update conversation history
            self.conversation_history.append({
                "user": query_text, 
                "system": web_result["answer"]
            })
            
            # Optional: Add web search results to vector store for future queries
            # self._add_web_results_to_vectorstore(query_text)
            
            logger.info(f"Total query processing time: {time.time() - start_time:.2f} seconds")
            return web_result
            
        except Exception as e:
            logger.error(f"Error during enhanced query processing: {str(e)}")
            return {
                "error": str(e),
                "answer": f"I encountered an error while processing your query: {str(e)}",
                "sources": [],
                "query_time": time.time() - start_time,
                "from_web": False
            }
    
    def _add_web_results_to_vectorstore(self, query_text: str, num_results: int = 3) -> None:
        """Optionally add web search results to the vector store for future use."""
        try:
            # Get documents from web search
            web_docs = self.web_search_agent.create_documents_from_search(query_text, num_results)
            
            if web_docs and self.vector_store:
                # Add to vector store
                texts = [doc.page_content for doc in web_docs]
                metadatas = [doc.metadata for doc in web_docs]
                self.vector_store.add_texts(texts, metadatas)
                logger.info(f"Added {len(web_docs)} web search results to vector store")
        except Exception as e:
            logger.error(f"Error adding web results to vector store: {str(e)}")

