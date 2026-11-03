"""Groq calls, error classification and output validation.

The client is created only when a request actually needs generation.
Raw provider errors are logged server-side and never shown in the chat.
"""
import logging
import re
import time
from dataclasses import dataclass

import config

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = (
    "You are an AI Health Symptoms Assistant for general education only.\n"
    "Use ONLY the numbered reference passages provided in the user message. Do not add facts "
    "from your own knowledge. If the passages do not cover part of the question, write exactly: "
    "'The reference documents do not cover this.' for that part.\n"
    "Never give a definite diagnosis, prescription, or medicine dosage. Use cautious wording "
    "like 'may be related to'.\n"
    "Cite the passage after every statement, like [S1] or [S2]. Only cite numbers that exist.\n"
    "Reply in exactly this format, with these four bold headings:\n"
    "**Possible common causes:**\n"
    "**Self-care tips:**\n"
    "**See a doctor if:**\n"
    "**Emergency signs:**\n"
    "Keep it short, simple, and calm. End with: 'Please consult a doctor for proper advice.'"
)

REQUIRED_SECTIONS = (
    "possible common causes",
    "self-care tips",
    "see a doctor if",
    "emergency signs",
)
_DOSAGE = re.compile(r"\b\d+(?:\.\d+)?\s?(?:mg|mcg|µg|ml|g|iu)\b", re.IGNORECASE)
_CITATION = re.compile(r"\[s(\d+)\]", re.IGNORECASE)


@dataclass
class LLMResult:
    ok: bool
    text: str = ""
    error: str = ""  # "missing_key" | "auth" | "temporary" | "empty"


def _make_client(api_key):
    from groq import Groq  # imported lazily so the app starts without the key or package

    return Groq(api_key=api_key)


def classify_error(exc):
    """'auth' for invalid/forbidden credentials, otherwise 'temporary'."""
    status = getattr(exc, "status_code", None)
    name = type(exc).__name__
    if name in ("AuthenticationError", "PermissionDeniedError") or status in (401, 403):
        return "auth"
    return "temporary"


def generate(messages, *, api_key, model, attempts=3, sleep=time.sleep, client_factory=_make_client):
    """Call the model with retries. Never raises; returns an LLMResult."""
    if not api_key:
        return LLMResult(False, error="missing_key")
    try:
        client = client_factory(api_key)
    except Exception as exc:
        logger.warning("LLM client setup failed: %s", type(exc).__name__)
        return LLMResult(False, error=classify_error(exc))

    last_error = "temporary"
    for attempt in range(1, attempts + 1):
        try:
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                max_tokens=config.MAX_TOKENS,
                temperature=0.2,
            )
            text = (response.choices[0].message.content or "").strip()
            if text:
                return LLMResult(True, text)
            last_error = "empty"
        except Exception as exc:
            kind = classify_error(exc)
            logger.warning("LLM call failed (%s): %s", kind, type(exc).__name__)
            if kind == "auth":
                return LLMResult(False, error="auth")
            last_error = kind
        if attempt < attempts:
            sleep(1.5 * attempt)
    return LLMResult(False, error=last_error)


def cited_indices(text):
    return {int(n) for n in _CITATION.findall(text or "")}


def validate_answer(text, source_count):
    """Return a list of problems. Empty list means the answer passes the checks."""
    stripped = (text or "").strip()
    if not stripped:
        return ["answer is blank"]
    low = stripped.lower()
    problems = [f"missing section: {s}" for s in REQUIRED_SECTIONS if s not in low]
    if "consult a doctor" not in low:
        problems.append("missing closing limitation")
    if source_count:
        cites = cited_indices(stripped)
        if not cites:
            problems.append("no citations")
        elif any(c < 1 or c > source_count for c in cites):
            problems.append("citation to a source that does not exist")
    if _DOSAGE.search(stripped):
        problems.append("contains a possible medicine dose")
    return problems
