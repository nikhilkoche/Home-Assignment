from dotenv import load_dotenv
import os

load_dotenv()

APP_CONFIG: dict = {
    'host': '0.0.0.0',
    'port': 8000,
    'llm_model': 'gpt-4o-mini',
    'llm_temperature': 0.5,
    'pinecone_key': os.getenv('PINECONE_API_KEY'),
    'openai_key': os.getenv('OPENAI_API_KEY'),
}