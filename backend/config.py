import os
from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULT_CSV = os.path.join(PROJECT_ROOT, "data", "uploads", "csv", "Resume.csv")
CSV_PATH = os.getenv("CSV_PATH", DEFAULT_CSV)

CORP_EMBED_BASE_URL = os.getenv(
    "CORP_EMBED_BASE_URL"
).rstrip("/")

CORP_API_KEY = os.getenv("CORP_API_KEY", "").strip()
CORP_EMBED_PROVIDER = os.getenv("CORP_EMBED_PROVIDER", "sagemaker").strip()
CORP_EMBED_MODEL = os.getenv("CORP_EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2").strip()

EMBED_BATCH = int(os.getenv("EMBED_BATCH", "128"))
HTTP_TIMEOUT = float(os.getenv("HTTP_TIMEOUT", "60"))
VERIFY_SSL = os.getenv("VERIFY_SSL", "true").lower() == "true"
RETRY_TIMES = int(os.getenv("RETRY_TIMES", "3"))