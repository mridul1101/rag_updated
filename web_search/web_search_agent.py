from typing import Dict, Any, List, Optional
import logging
import time
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.schema import Document
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain

# Configure logging
logger = logging.getLogger(__name__)

class WebSearchAgent:
    def __init__(self, api_key: str, model_name: str = "gemini-2.0-flash"):
        """Initialize the web search agent with API key and model."""
        self.api_key = api_key
        self.model_name = model_name
        self.search_engine = DuckDuckGoSearchAPIWrapper()
        
        # Initialize Gemini model
        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            streaming=True,
            callbacks=[StreamingStdOutCallbackHandler()]
        )
        
        # Create synthesis prompt template
        self.synthesis_prompt = PromptTemplate(
            input_variables=["query", "search_results"],
            template="""
            You are an AI assistant helping to find information from web searches.
            
            USER QUERY: {query}
            
            WEB SEARCH RESULTS:
            {search_results}
            
            Instructions:
            1. Synthesize a comprehensive answer based on the web search results above.
            2. Only include information that is directly relevant to the user's query.
            3. Cite your sources using [Source X] notation where X is the number of the search result.
            4. If the search results don't contain a clear answer, acknowledge this and provide the most helpful response possible.
            5. Format your answer in clear, concise language.
            
            YOUR RESPONSE:
            """
        )
        
        # Create synthesis chain
        self.synthesis_chain = LLMChain(
            llm=self.llm,
            prompt=self.synthesis_prompt
        )
        
        logger.info(f"Web Search Agent initialized with model {model_name}")
    
    def search(self, query: str, num_results: int = 5) -> List[Dict[str, str]]:
        """Perform a web search and return results."""
        logger.info(f"Performing web search for: {query}")
        start_time = time.time()
        
        try:
            search_results = self.search_engine.results(query, num_results)
            logger.info(f"Search completed in {time.time() - start_time:.2f} seconds with {len(search_results)} results")
            
            formatted_results = []
            for i, result in enumerate(search_results, 1):
                formatted_results.append({
                    "index": i,
                    "title": result.get("title", "No title"),
                    "link": result.get("link", "No link"),
                    "snippet": result.get("snippet", "No snippet")
                })
            
            return formatted_results
        except Exception as e:
            logger.error(f"Search error: {str(e)}")
            return []
    
    def synthesize_answer(self, query: str, search_results: List[Dict[str, str]]) -> str:
        """Synthesize search results into a coherent answer."""
        if not search_results:
            return "I couldn't find any relevant information from web searches."
        
        logger.info(f"Synthesizing answer from {len(search_results)} search results")
        start_time = time.time()
        
        # Format search results for prompt
        formatted_results = ""
        for result in search_results:
            formatted_results += f"[Source {result['index']}]\n"
            formatted_results += f"Title: {result['title']}\n"
            formatted_results += f"URL: {result['link']}\n" 
            formatted_results += f"Snippet: {result['snippet']}\n\n"
        
        try:
            # Generate answer
            response = self.synthesis_chain.invoke({
                "query": query,
                "search_results": formatted_results
            })
            
            logger.info(f"Answer synthesized in {time.time() - start_time:.2f} seconds")
            return response["text"]
        except Exception as e:
            logger.error(f"Synthesis error: {str(e)}")
            return f"I found some information but encountered an error synthesizing the answer: {str(e)}"
    
    def query(self, query_text: str) -> Dict[str, Any]:
        """Process a query through web search and return formatted response."""
        logger.info(f"Processing web search query: {query_text}")
        start_time = time.time()
        
        try:
            # Perform search
            search_results = self.search(query_text)
            
            if not search_results:
                return {
                    "answer": "I couldn't find any relevant information from web searches.",
                    "sources": [],
                    "query_time": time.time() - start_time
                }
            
            # Synthesize answer
            answer = self.synthesize_answer(query_text, search_results)
            
            # Format source information for return
            sources = []
            for result in search_results:
                sources.append({
                    "content": result["snippet"],
                    "metadata": {
                        "document_name": f"Web: {result['title']}",
                        "source_url": result["link"],
                        "source_index": result["index"]
                    }
                })
            
            result = {
                "answer": answer,
                "sources": sources,
                "query_time": time.time() - start_time,
                "from_web": True
            }
            
            logger.info(f"Web query completed in {result['query_time']:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Error during web query: {str(e)}")
            return {
                "error": str(e),
                "answer": f"I encountered an error while searching the web: {str(e)}",
                "sources": [],
                "query_time": time.time() - start_time,
                "from_web": True
            }

    def create_documents_from_search(self, query: str, num_results: int = 5) -> List[Document]:
        """Convert search results to Document objects for potential indexing."""
        search_results = self.search(query, num_results)
        documents = []
        
        for result in search_results:
            doc = Document(
                page_content=f"{result['title']}\n\n{result['snippet']}",
                metadata={
                    "source": "web_search",
                    "url": result["link"],
                    "title": result["title"],
                    "document_name": f"Web: {result['title']}"
                }
            )
            documents.append(doc)
        
        return documents