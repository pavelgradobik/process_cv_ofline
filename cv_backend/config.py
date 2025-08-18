import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "Resume.csv")
CSV_PATH = os.getenv("CSV_PATH", DEFAULT_CSV)

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
TFIDF_MAX_FEATURES = int(os.getenv("TFIDF_MAX_FEATURES", "50000"))

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