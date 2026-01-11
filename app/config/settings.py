from pathlib import Path


# Confidence gating
CONFIDENCE_THRESHOLD = 0.71


BASE_DIR = Path(__file__).resolve().parents[2]

VECTOR_STORE_PATH = BASE_DIR / "vector_store"
UPLOAD_DIR = BASE_DIR / "data" / "uploads"

# Ensure upload directory exists
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

CONFIDENCE_THRESHOLD = 0.65
