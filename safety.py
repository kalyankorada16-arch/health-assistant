"""Local emergency screening. Runs with no API key and no network.

This is a reviewed keyword-and-pattern policy with negation handling, NOT a
complete triage system. It errs toward escalating when meaning is unclear.
Known limits: English only; no analysis of tense or who the symptom is about
("my father had a stroke last year" still escalates).
"""
import re
import unicodedata
from dataclasses import dataclass, field

import config

_CHAR_MAP = str.maketrans({
    "’": "'", "‘": "'", "ʼ": "'", "´": "'", "`": "'",
    "“": '"', "”": '"',
})


def normalize(text):
    """Lowercase, unify apostrophes, expand contractions, strip odd punctuation."""
    t = unicodedata.normalize("NFKC", text or "").translate(_CHAR_MAP).lower()
    t = t.replace("-", " ").replace("_", " ")
    t = re.sub(r"\bcan ?not\b|\bcan't\b|\bcant\b", "cannot", t)
    t = re.sub(r"\bwon't\b", "will not", t)
    t = re.sub(r"\bdont\b", "do not", t)
    t = re.sub(r"\b(\w+)n't\b", r"\1 not", t)
    t = re.sub(r"[^a-z0-9\s.,;!?]", " ", t)
    return re.sub(r"\s+", " ", t).strip()


URGENT = {
    "breathing": [
        r"\bcannot (?:\w+ ){0,2}breath\w*",
        r"\b(?:struggl\w*|trouble|difficult\w*|hard|gasp\w*|unable|hardly able|can hardly|barely able|not able)\b(?: \w+){0,3} breath\w*",
        r"\bshort(?:ness)? of breath\b",
        r"\bbreathless\w*",
        r"\bchoking\b",
        r"\bstopped breathing\b",
    ],
    "chest": [
        r"\bchest (?:pain|pains|tightness|pressure|discomfort|heaviness)\b",
        r"\b(?:pain|pressure|tightness|heaviness|discomfort) (?:in|on|across) (?:my |the |your )?chest\b",
        r"\btight chest\b",
        r"\bheart attack\b",
    ],
    "stroke": [
        r"\bstroke\b",
        r"\bface (?:is )?droop\w*",
        r"\bdroop\w* (?:face|mouth|eyelid)",
        r"\bslurr\w*",
        r"\bsudden(?:ly)? (?:\w+ ){0,2}(?:weak\w*|numb\w*|confus\w*|vision loss|trouble speaking)",
        r"\bcannot (?:speak|talk|move (?:my )?(?:arm|leg|face))",
    ],
    "bleeding": [
        r"\bsevere(?:ly)? bleeding\b",
        r"\bheavy bleeding\b",
        r"\bbleeding (?:heavily|badly|a lot|profusely|will not stop|does not stop)",
        r"\bcannot stop (?:the )?bleeding",
        r"\b(?:cough\w*|vomit\w*|throw\w*|spit\w*)(?: up)?(?: \w+)? blood\b",
    ],
    "consciousness": [
        r"\bunconscious\b",
        r"\bpassed out\b",
        r"\bfaint(?:ed|ing)\b",
        r"\bblack(?:ed|ing) out\b",
        r"\bunresponsive\b|\bnot responding\b|\bnot waking up\b",
        r"\bseizure\w*|\bconvuls\w*|\bfitting\b",
    ],
    "poison": [
        r"\boverdos\w*",
        r"\bpoison\w*",
        r"\bsnake ?bite\w*|\bbitten by (?:a |an )?(?:snake|scorpion)\b",
        r"\b(?:swallow\w*|drank|drink\w*|ate|ingest\w*) (?:\w+ ){0,3}(?:bleach|pesticide|kerosene|acid|detergent|insecticide|rat killer)\b",
        r"\btook too many (?:pills|tablets|capsules)\b",
    ],
    "allergic": [
        r"\banaphyla\w*",
        r"\bsevere allergic\b",
        r"\bswelling (?:of|in) (?:my |the )?(?:throat|tongue)\b",
        r"\b(?:throat|tongue) (?:is |feels? )?(?:swelling|swollen|closing)\b",
        r"\bthroat closing\b",
    ],
    "neuro": [
        r"\bworst headache\b|\bthunderclap\b",
    ],
    "selfharm": [
        r"\bsuicid\w*",
        r"\bkill(?:ing)? myself\b",
        r"\bend(?:ing)? my (?:own )?life\b",
        r"\bwant(?:s)? to die\b|\bwanna die\b",
        r"\b(?:do not|no longer) want to (?:live|be alive|be here)\b",
        r"\b(?:hurt|harm|cut)(?:ting)? myself\b",
        r"\bself harm\w*",
        r"\bbetter off dead\b",
    ],
}

# Needs a second symptom present (stiff neck alone is common and only needs a check).
STIFF_NECK = r"\bstiff neck\b"
STIFF_NECK_PARTNERS = r"\bfever\b|\bhigh temperature\b|\bvomit\w*|\brash\b|\bsensitive to light\b|\bheadache\b"

CLARIFY = [
    STIFF_NECK,
    r"\bchest\b(?: \w+){0,3} (?:hurts?|aches?|aching|heavy|funny|burning)\b",
    r"\bpalpitation\w*|\bheart (?:is )?(?:racing|pounding|fluttering|skipping)",
    r"\b(?:dizz\w+|light ?headed)\b",
    r"\bblood in (?:my )?(?:urine|stool|spit|phlegm|cough)\b",
    r"\bconfus\w+",
    r"\bwheez\w+",
    r"\b(?:lips|face) (?:is |are )?(?:swelling|swollen)\b",
    r"\bsevere (?:headache|pain|abdominal pain|stomach pain)\b",
    r"\bblurred vision\b|\bdouble vision\b",
]

_BOUNDARY = re.compile(
    r"[.;!?]|,?\s+(?:but|and|only|however|except|although|though|because|so|now|still)\s+|,\s+(?:i|my|it|the)\s"
)
_NEGATION = re.compile(r"\b(?:no|not|without|never|denies|denied|none|nothing|free of|negative for)\b")
_UNCERTAIN = re.compile(
    r"\b(?:not sure|unsure|do not know|wonder\w*|if i have|if i am having|if this is|could (?:it|this) be)\b"
)

_WINDOW = 6
_WINDOW_SELFHARM = 3  # keep negation tight: "I have no one and I want to die" must escalate

_AFFIRMATIVE = {
    "yes", "yeah", "yep", "yup", "y", "yes it is", "yes i do", "yes it does",
    "yes it is happening", "yes now", "yes right now", "it is", "it is happening now",
}


@dataclass
class SafetyResult:
    level: str  # "emergency" | "clarify" | "none"
    categories: list = field(default_factory=list)
    matched: list = field(default_factory=list)
    message: str = ""


def _status(text, start, window):
    """Classify a match as 'present', 'negated' or 'uncertain' from the words before it."""
    prefix = text[:start]
    cut = 0
    for m in _BOUNDARY.finditer(prefix):
        cut = m.end()
    words = prefix[cut:].split()[-window:]
    context = " ".join(words)
    if _UNCERTAIN.search(context):
        return "uncertain"
    if _NEGATION.search(context):
        return "negated"
    return "present"


def _scan(text, patterns, window):
    """Return a list of (status, matched_text) for every pattern hit."""
    hits = []
    for pattern in patterns:
        for m in re.finditer(pattern, text):
            hits.append((_status(text, m.start(), window), m.group(0)))
    return hits


def check(text):
    """Screen one message. Pure function of the text: safe to call with no API key."""
    t = normalize(text)
    if not t:
        return SafetyResult("none")

    emergency, uncertain, matched = [], [], []
    for category, patterns in URGENT.items():
        window = _WINDOW_SELFHARM if category == "selfharm" else _WINDOW
        for status, phrase in _scan(t, patterns, window):
            if status == "present":
                if category not in emergency:
                    emergency.append(category)
                matched.append(phrase)
            elif status == "uncertain":
                uncertain.append(phrase)

    neck = _scan(t, [STIFF_NECK], _WINDOW)
    if any(s == "present" for s, _ in neck):
        partners = _scan(t, [STIFF_NECK_PARTNERS], _WINDOW)
        if any(s == "present" for s, _ in partners):
            if "neuro" not in emergency:
                emergency.append("neuro")
            matched.append("stiff neck")

    if emergency:
        return SafetyResult("emergency", emergency, matched, emergency_message(emergency))

    soft = uncertain[:]
    for status, phrase in _scan(t, CLARIFY, _WINDOW):
        if status in ("present", "uncertain"):
            soft.append(phrase)
    if soft:
        return SafetyResult("clarify", [], soft, clarify_message())

    return SafetyResult("none")


def is_affirmative(text):
    """True if a short reply to a clarification question means 'yes, it is urgent'."""
    return normalize(text).strip(" .!?,") in _AFFIRMATIVE


_LABELS = {
    "breathing": "trouble breathing",
    "chest": "chest pain or pressure",
    "stroke": "possible stroke signs",
    "bleeding": "serious bleeding",
    "consciousness": "fainting, seizure or unresponsiveness",
    "poison": "poisoning, overdose or a venomous bite",
    "allergic": "a severe allergic reaction",
    "neuro": "severe headache or stiff neck with other symptoms",
    "general": "a possible emergency",
}


def emergency_message(categories):
    numbers = f"**{config.EMERGENCY_NUMBER}**"
    if config.AMBULANCE_NUMBER:
        numbers += f" (ambulance: **{config.AMBULANCE_NUMBER}**)"

    physical = [c for c in categories if c != "selfharm"]
    parts = []
    if physical:
        labels = ", ".join(_LABELS.get(c, "a possible emergency") for c in physical)
        parts.append(
            f"🚨 **Your message mentions possible emergency symptoms: {labels}.**\n\n"
            f"Call {numbers} or go to the nearest emergency department now. "
            "Do not wait for an online answer. If you can, ask someone nearby to stay with you."
        )
    if "selfharm" in categories:
        parts.append(
            "💙 **If you are thinking about harming yourself or ending your life, you do not have to face it alone.**\n\n"
            f"If you are in immediate danger, call **{config.EMERGENCY_NUMBER}** now. "
            f"You can also call the Tele-MANAS mental health helpline on **{config.MENTAL_HEALTH_HELPLINE}**, "
            "or tell someone you trust right now and ask them to stay with you."
        )
    return "\n\n".join(parts)


def clarify_message():
    return (
        "I want to make sure I understand. You mentioned a symptom that can sometimes be serious. "
        "Is it **happening right now, sudden, or severe**? Reply **yes** or **no**, or tell me more.\n\n"
        f"**If you think this could be an emergency, call {config.EMERGENCY_NUMBER} now instead of waiting.**"
    )
