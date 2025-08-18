import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "uploads", "csv", "Resume.csv")
GENERATIVE_ENGINE_API_KEY = os.getenv("GENERATIVE_ENGINE_API_KEY", "")
GENERATIVE_ENGINE_BASE_URL = os.getenv("GENERATIVE_ENGINE_BASE_URL", "").rstrip("/")
EMBEDDING_MODEL = os.getenv("GENERATIVE_ENGINE_EMBEDDING_MODEL", "text-embedding-3-small")
CHAT_MODEL = os.getenv("GENERATIVE_ENGINE_CHAT_MODEL", "openai.gpt-3.5-turbo")
REQUEST_TIMEOUT = float(os.getenv("GENERATIVE_ENGINE_TIMEOUT_SEC", "240"))
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
USE_ENGINE_EMBEDDINGS = os.getenv("USE_ENGINE_EMBEDDINGS", "False").lower() == "False"
CSV_PATH = os.getenv("CSV_PATH", DEFAULT_CSV)

ROLE_SYNONYMS = {
    "designer": [
        "designer", "product designer", "ux designer", "ui designer",
        "ux/ui designer", "ui/ux designer", "graphic designer",
        "visual designer", "interaction designer", "ux researcher",
        "service designer"
    ],
    "python developer": [
        "python developer", "python engineer", "backend python developer",
        "software engineer python", "data engineer (python)",
        "ml engineer (python)"
    ],
    "qa": ["qa", "qa engineer", "test engineer", "quality assurance"],
    "frontend": ["frontend developer", "frontend engineer", "react developer", "vue developer"],
    "backend": ["backend developer", "backend engineer", "server-side developer"],
}
