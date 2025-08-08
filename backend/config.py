import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV=os.path.join(PROJECT_ROOT,"data", "uploads", "csv", "Resume.csv")

GENERATIVE_ENGINE_API_KEY = os.getenv("GENERATIVE_ENGINE_API_KEY")
GENERATIVE_ENGINE_BASE_URL = os.getenv("GENERATIVE_ENGINE_BASE_URL")
GENERATIVE_ENGINE_MODEL = os.getenv("GENERATIVE_ENGINE_MODEL", "openai.gpt-4o")
CSV_PATH = os.getenv("CSV_PATH", DEFAULT_CSV)
