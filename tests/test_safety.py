import pytest

import safety


@pytest.mark.parametrize("text", [
    "I have chest pain",
    "I can’t breathe",          # curly apostrophe
    "I can't breathe",
    "I cant breathe",
    "I am struggling to breathe",
    "having trouble breathing",
    "shortness of breath since morning",
    "my father is having a stroke",
    "he is coughing up blood",
    "she passed out and is unresponsive",
    "I took too many pills",
    "bitten by a snake",
    "my throat is closing",
    "worst headache of my life",
    "I have a stiff neck and a fever",
    "I want to die",
    "I have no one and I want to die",
    "I want to kill myself",
])
def test_emergency_phrases_and_paraphrases(text):
    assert safety.check(text).level == "emergency", text


@pytest.mark.parametrize("text", [
    "I have no chest pain, only a runny nose",
    "no chest pain or shortness of breath, just a cough",
    "I don't have any trouble breathing",
    "I did not faint, I just have a headache",
    "no trouble breathing and no fever",
    "I do not want to die, I just feel tired",
    "mild sore throat and runny nose for 2 days",
])
def test_negation_not_escalated(text):
    assert safety.check(text).level in ("none", "clarify"), text
    assert safety.check(text).level != "emergency", text


def test_negation_does_not_hide_a_later_symptom():
    assert safety.check("no fever but I cannot breathe").level == "emergency"
    assert safety.check("I have no fever and my chest pain is crushing").level == "emergency"


def test_uncertain_wording_asks_for_clarification():
    r = safety.check("I am not sure if I have chest pain")
    assert r.level == "clarify"
    assert "112" in r.message


def test_soft_symptoms_ask_for_clarification():
    assert safety.check("I feel dizzy since morning").level == "clarify"
    assert safety.check("I have a stiff neck").level == "clarify"


def test_ordinary_symptoms_are_not_flagged():
    assert safety.check("fever and sore throat for 2 days").level == "none"
    assert safety.check("").level == "none"
    assert safety.check("   ").level == "none"


def test_normalize_handles_punctuation_and_contractions():
    assert safety.normalize("I Can’t   BREATHE!!") == "i cannot breathe!!"
    assert "do not" in safety.normalize("I don't know")


def test_self_harm_message_is_not_shown_for_other_emergencies():
    chest = safety.check("I have chest pain")
    assert "Tele-MANAS" not in chest.message
    harm = safety.check("I want to kill myself")
    assert "Tele-MANAS" in harm.message
    assert "ambulance" not in harm.message.lower()


def test_both_messages_when_both_apply():
    r = safety.check("I have chest pain and I want to die")
    assert "Tele-MANAS" in r.message and "emergency symptoms" in r.message


def test_affirmative_reply_to_clarification():
    assert safety.is_affirmative("Yes")
    assert safety.is_affirmative("yeah.")
    assert not safety.is_affirmative("no")
    assert not safety.is_affirmative("yes but it stopped yesterday and I feel fine now")


def test_works_without_any_api_configuration(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    assert safety.check("I can't breathe").level == "emergency"
