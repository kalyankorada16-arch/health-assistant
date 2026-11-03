"""Orchestrates one question: local safety check -> retrieval -> generation -> validation.

No Streamlit imports, so everything here is unit-testable. The order matters:
escalation and the 'not covered' path never depend on the API being available.
"""
import re
from dataclasses import dataclass, field

import config
import llm_service
import safety

AGE_UNITS = ("years", "months", "weeks")


@dataclass
class Profile:
    age_value: float
    age_unit: str  # one of AGE_UNITS
    sex: str
    duration: str
    severity: int

    def age_in_months(self):
        if self.age_unit == "years":
            return self.age_value * 12
        if self.age_unit == "months":
            return self.age_value
        return self.age_value / 4.345

    def describe(self):
        return (
            f"Age: {self.age_value:g} {self.age_unit}; Sex: {self.sex}; "
            f"Duration: {self.duration}; Severity (1-10): {self.severity}"
        )

    def to_dict(self):
        return {
            "age_value": self.age_value, "age_unit": self.age_unit, "sex": self.sex,
            "duration": self.duration, "severity": self.severity,
        }

    @classmethod
    def from_dict(cls, d):
        return cls(d["age_value"], d["age_unit"], d["sex"], d["duration"], d["severity"])


@dataclass
class Reply:
    content: str
    kind: str  # emergency | clarify | scope | unsupported | no_key | error | answer
    sources: list = field(default_factory=list)
    problems: list = field(default_factory=list)


def _finish(reply):
    """Every reply, including errors and escalations, carries the disclaimer."""
    reply.content = f"{reply.content}\n\n_{config.DISCLAIMER_SHORT}_"
    return reply


def _excerpt(text, max_words=70):
    words = text.split()
    return " ".join(words[:max_words]) + (" ..." if len(words) > max_words else "")


def _source_list(hits):
    sources = []
    for i, hit in enumerate(hits, start=1):
        d = hit.chunk.doc
        sources.append({
            "label": f"S{i}", "title": d.title, "publisher": d.publisher, "url": d.url,
            "published": d.published, "reviewed": d.reviewed, "version": d.version,
            "excerpt": _excerpt(hit.chunk.text), "text": hit.chunk.text,
        })
    return sources


def _public_sources(sources):
    """Source dicts without the full chunk text, for display and storage."""
    return [{k: v for k, v in s.items() if k != "text"} for s in sources]


def _extractive(sources):
    lines = ["Here are the most relevant passages from the reference documents:"]
    for s in sources:
        lines.append(f"> {s['excerpt']} **[{s['label']}]**")
    return "\n\n".join(lines)


def _retrieval_query(question, history):
    """Short follow-ups ('It started yesterday') borrow the previous symptom message."""
    if len(question.split()) <= 8:
        for m in reversed(history):
            if m.get("role") == "user":
                return f"{m['content']} {question}"
    return question


def _history_messages(history):
    recent = [m for m in history if m["role"] == "user" or m.get("kind") == "answer"]
    recent = recent[-config.MAX_HISTORY_MESSAGES:]
    messages = []
    for m in recent:
        if m["role"] == "user":
            details = Profile.from_dict(m["profile"]).describe() if m.get("profile") else "Details not recorded"
            messages.append({"role": "user", "content": f"[Patient details at that time: {details}]\nSymptoms: {m['content']}"})
        else:
            messages.append({"role": "assistant", "content": m["content"]})
    return messages


def build_messages(question, profile, history, sources):
    passages = "\n\n".join(f"[{s['label']}] ({s['title']}, {s['publisher']})\n{s['text']}" for s in sources)
    current = (
        f"Reference passages:\n{passages}\n\n"
        f"Current patient details: {profile.describe()}\n"
        f"Symptoms / question: {question}"
    )
    return [{"role": "system", "content": llm_service.SYSTEM_PROMPT},
            *_history_messages(history),
            {"role": "user", "content": current}]


def answer_question(question, profile, history, retriever, *, api_key=None, model=None,
                    llm=llm_service.generate):
    """Return a Reply for one user message, or None if the message is blank.

    `history` is the list of earlier chat entries (not including this question).
    """
    question = (question or "").strip()
    if not question:
        return None

    # 1. Local escalation: needs no API key, no network, no documents.
    last_assistant = next((m for m in reversed(history) if m["role"] == "assistant"), None)
    if last_assistant and last_assistant.get("kind") == "clarify" and safety.is_affirmative(question):
        return _finish(Reply(safety.emergency_message(["general"]), "emergency"))
    result = safety.check(question)
    if result.level == "emergency":
        return _finish(Reply(result.message, "emergency"))
    if result.level == "clarify":
        return _finish(Reply(result.message, "clarify"))

    # 2. Scope: this tool does not support infants under 12 months.
    if profile.age_in_months() < 12:
        return _finish(Reply(
            "This assistant is not designed for infants under 12 months. Babies can become unwell "
            "quickly, so please contact a paediatrician or your local health service about any "
            f"symptoms. If your baby is struggling to breathe, is hard to wake, or you are worried, call {config.EMERGENCY_NUMBER}.",
            "scope"))

    # 3. Retrieval: no supporting passage means no model-knowledge answer.
    hits = retriever.search(_retrieval_query(question, history), k=config.TOP_K,
                            min_score=config.MIN_SCORE) if retriever else []
    if not hits:
        return _finish(Reply(
            "The reference documents available to me do not cover this question, so I can't give "
            "a reliable answer. Please speak to a doctor or pharmacist about it.", "unsupported"))
    sources = _source_list(hits)
    shown = _public_sources(sources)

    # 4. Generation (only now is the API needed).
    if not api_key:
        return _finish(Reply(
            "Answer generation is not configured right now. " + _extractive(sources),
            "no_key", shown))

    messages = build_messages(question, profile, history, sources)
    result = llm(messages, api_key=api_key, model=model or config.get_model_name())
    if not result.ok:
        if result.error == "auth":
            text = "The AI service could not be used because its credentials are not valid. The site owner needs to fix this."
        else:
            text = "The AI service is temporarily unavailable. Please try again in a moment."
        return _finish(Reply(f"{text}\n\n{_extractive(sources)}", "error", shown))

    # 5. Validate; retry once with a correction; otherwise fall back to the passages.
    problems = llm_service.validate_answer(result.text, len(sources))
    if problems:
        retry_messages = messages + [
            {"role": "assistant", "content": result.text},
            {"role": "user", "content": "Your answer broke the rules: " + "; ".join(problems)
             + ". Rewrite it using only the passages, the four headings, [S#] citations, and the closing sentence."},
        ]
        retry = llm(retry_messages, api_key=api_key, model=model or config.get_model_name())
        if retry.ok:
            problems = llm_service.validate_answer(retry.text, len(sources))
            if not problems:
                result = retry
    if problems:
        return _finish(Reply(
            "I couldn't produce a reliable summary this time. " + _extractive(sources),
            "error", shown, problems))

    text = result.text.strip()
    cited = llm_service.cited_indices(text)
    shown = [s for s in shown if int(s["label"][1:]) in cited]
    if profile.severity >= 9:
        text += "\n\n⚠️ You rated your symptoms very severe. Please consider seeing a doctor today."
    return _finish(Reply(text, "answer", shown))
