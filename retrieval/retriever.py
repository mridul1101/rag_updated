from typing import List
import re
from langchain.schema import Document
from langchain.vectorstores import VectorStore
from pydantic import BaseModel, Field,ConfigDict
from config.config import logger
from langchain.schema import BaseRetriever
class DateFilteredRetriever(BaseRetriever,BaseModel):
    vector_store: VectorStore = Field(...)
    k: int = Field(default=4)

    # Allow arbitrary types (e.g., VectorStore)
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def get_relevant_documents(self, query: str) -> List[Document]:
        date = self.extract_date_from_query(query)
        all_docs = self.vector_store.similarity_search(query, k=self.k * 2)
        filtered_docs = [doc for doc in all_docs if doc.metadata.get("date", "") == date]

        if not filtered_docs:
            logger.warning(f"No documents found for date {date}. Returning most similar documents.")
            return all_docs[:self.k]

        return filtered_docs[:self.k]

    def extract_date_from_query(self, query: str) -> str:
        """Extract date from query (e.g., 'April 2024')."""
        date_pattern = r"(January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{4}"
        match = re.search(date_pattern, query, re.IGNORECASE)
        if match:
            return match.group(0)
        return ""
    
