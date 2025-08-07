import os
from dotenv import load_dotenv

load_dotenv()

GENERATIVE_ENGINE_API_KEY = os.getenv("GENERATIVE_ENGINE_API_KEY")
GENERATIVE_ENGINE_BASE_URL = os.getenv("GENERATIVE_ENGINE_BASE_URL")
GENERATIVE_ENGINE_MODEL = os.getenv("GENERATIVE_ENGINE_MODEL", "openai.gpt-4o")
CSV_PATH = os.getenv("CSV_PATH", "data/uploads/csv/Resume.csv")
