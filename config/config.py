import os

# Configure logging
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Constants
API_KEY = "AIzaSyCPFn4GA_6HAAe4BkFGXGfHUJmGs5xjFKQ"
MODEL_NAME = "gemini-2.0-flash"
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

INDEX_PATH = "my_faiss_index"
