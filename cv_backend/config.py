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
    "Java Developer": [
        "java engineer", "backend java developer", "core java developer", "j2ee developer"
    ],
    "Testing": [
        "qa", "quality assurance", "software tester", "test engineer", "sdet", "automation tester"
    ],
    "DevOps Engineer": [
        "devops", "site reliability engineer", "sre", "cloud engineer", "infrastructure engineer", "ci/cd engineer"
    ],
    "Python Developer": [
        "python engineer", "backend python developer", "django developer", "flask developer"
    ],
    "Web Designing": [
        "web designer", "ui designer", "frontend designer", "html css designer"
    ],
    "HR": [
        "human resources", "hr manager", "talent acquisition", "recruiter", "people operations"
    ],
    "Hadoop": [
        "big data engineer", "hadoop developer", "spark developer", "mapreduce engineer"
    ],
    "Blockchain": [
        "blockchain engineer", "solidity developer", "smart contract developer", "web3 developer"
    ],
    "ETL Developer": [
        "etl engineer", "data engineer", "sql etl developer", "informatica developer"
    ],
    "Operations Manager": [
        "ops manager", "operations lead", "business operations manager"
    ],
    "Data Science": [
        "data scientist", "machine learning engineer", "ml engineer", "ai engineer"
    ],
    "Sales": [
        "sales executive", "business development", "account executive", "sales representative"
    ],
    "Mechanical Engineer": [
        "mechanical design engineer", "automotive engineer", "production engineer"
    ],
    "Electrical Engineering": [
        "electrical engineer", "electronics engineer", "power systems engineer"
    ],
    "Arts": [
        "artist", "fine arts", "graphic artist"
    ],
    "Database": [
        "database administrator", "dba", "sql developer", "database engineer"
    ],
    "Civil Engineer": [
        "civil engineering", "structural engineer", "construction engineer"
    ],
    "Advocate": [
        "lawyer", "legal counsel", "attorney"
    ],
    "Business Analyst": [
        "ba", "product analyst", "systems analyst"
    ],
    "Designer": [
        "graphic designer", "ui designer", "ux designer", "product designer"
    ],
    "Health and fitness": [
        "fitness trainer", "personal trainer", "gym instructor"
    ],
    "JavaScript Developer": [
        "js developer", "frontend developer", "react developer", "angular developer", "nodejs developer"
    ],
    "Automation Testing": [
        "qa automation", "automation tester", "sdet", "test automation engineer"
    ],
    "SAP Developer": [
        "sap engineer", "sap consultant", "sap abap developer"
    ],
    "Project Manager": [
        "pm", "technical project manager", "program manager", "delivery manager"
    ],
    "PMO": [
        "project management office", "pmo analyst", "pmo coordinator"
    ],
    "Electrical Engineer": [
        "electrical engineering", "electronics engineer", "power engineer"
    ],
    "Chartered Accountant": [
        "ca", "accountant", "financial auditor"
    ],
    "Business Development": [
        "bd", "sales development", "biz dev", "partnerships manager"
    ],
    "Civil Engineering": [
        "civil engineer", "construction engineer", "structural engineer"
    ],
}