import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Path to CSV (you can override with CSV_PATH env)
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "Resume.csv")
CSV_PATH = os.getenv("CSV_PATH", DEFAULT_CSV)

#Corporate engine params
GENERATIVE_ENGINE_API_KEY = os.getenv("GENERATIVE_ENGINE_API_KEY")
GENERATIVE_ENGINE_BASE_URL = os.getenv("GENERATIVE_ENGINE_BASE_URL")
GENERATIVE_ENGINE_MODEL = os.getenv("GENERATIVE_ENGINE_MODEL")
REQUEST_TIMEOUT = 240

# TF-IDF embedder params
EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "50000"))

# Role synonyms used for better recall in queries
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

    # NEW: broaden HR-related queries like "HR 20+ years"
    "hr": [
        "hr", "human resources", "hr manager", "hrbp", "hr business partner",
        "talent acquisition", "recruiter", "technical recruiter",
        "people operations", "people ops", "hr generalist", "hr specialist",
        "talent partner", "sourcer", "talent sourcer"
    ],
}