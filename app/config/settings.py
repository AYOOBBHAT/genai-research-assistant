from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]
VECTOR_STORE_PATH = BASE_DIR / "vector_store"


# Confidence gating
CONFIDENCE_THRESHOLD = 0.71
