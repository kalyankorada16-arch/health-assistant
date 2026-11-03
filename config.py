"""Central configuration. No network calls and no Streamlit imports here,
so importing this module is always safe (even with no API key)."""
import os
from pathlib import Path

from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / ".env")

KNOWLEDGE_DIR = BASE_DIR / "knowledge"
DEFAULT_MODEL = "llama-3.3-70b-versatile"

# Intended deployment region. VERIFY these numbers before every deployment
# and override them with environment variables if the region changes.
REGION = os.getenv("REGION", "India")
EMERGENCY_NUMBER = os.getenv("EMERGENCY_NUMBER", "112")  # national emergency number
AMBULANCE_NUMBER = os.getenv("AMBULANCE_NUMBER", "108")  # ambulance (many Indian states)
MENTAL_HEALTH_HELPLINE = os.getenv("MENTAL_HEALTH_HELPLINE", "14416")  # Tele-MANAS

# Retrieval / conversation limits
TOP_K = 4
MIN_SCORE = 0.12
MAX_HISTORY_MESSAGES = 6
MAX_STORED_MESSAGES = 40
MAX_TOKENS = 700

DISCLAIMER = (
    "**Educational information only.** This assistant is not a doctor and cannot "
    "diagnose, prescribe, or give medicine doses. Answers come only from the reference "
    f"documents listed under each reply. Emergency numbers shown are for **{REGION}**."
)
DISCLAIMER_SHORT = "Educational information only, not medical advice. Please consult a doctor."


def get_api_key():
    """Return the Groq API key, or None if missing or still a placeholder."""
    key = (os.getenv("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    if not key or key.lower().startswith("your_") or "your_key" in key.lower():
        return None
    return key


def get_model_name():
    """MODEL_NAME from the environment, else a sensible default. No network call."""
    return (os.getenv("MODEL_NAME") or "").strip() or DEFAULT_MODEL
