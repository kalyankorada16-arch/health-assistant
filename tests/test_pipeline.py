from pathlib import Path

import pytest

import config
import llm_service
import pipeline
from document_loader import load_chunks
from retriever import Retriever

FIXTURES = Path(__file__).parent / "fixtures"
GOOD = (
    "**Possible common causes:** A viral infection [S1].\n"
    "**Self-care tips:** Rest and fluids [S1].\n"
    "**See a doctor if:** It lasts over a week [S1].\n"
    "**Emergency signs:** Unable to swallow [S1].\n"
    "Please consult a doctor for proper advice."
)


@pytest.fixture(scope="module")
def retriever():
    chunks, _, _ = load_chunks(FIXTURES)
    return Retriever(chunks)


def profile(**kw):
    base = dict(age_value=25, age_unit="years", sex="Prefer not to say", duration="1-3 days", severity=4)
    base.update(kw)
    return pipeline.Profile(**base)


def fake_llm(*results):
    seen = []
    queue = list(results)

    def call(messages, **kwargs):
        seen.append(messages)
        return queue.pop(0)
    call.seen = seen
    return call


def ask(q, retriever, llm=None, history=None, api_key="k", **kw):
    return pipeline.answer_question(q, kw.pop("profile", profile()), history or [], retriever,
                                    api_key=api_key, model="m", llm=llm or fake_llm(), **kw)


def test_blank_question_returns_none(retriever):
    assert ask("   ", retriever) is None


def test_emergency_works_with_no_key_and_no_documents():
    r = ask("I can't breathe", None, api_key=None)
    assert r.kind == "emergency" and "112" in r.content


def test_emergency_never_calls_llm(retriever):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called")
    assert ask("I have chest pain", retriever, llm=boom).kind == "emergency"


def test_disclaimer_on_every_reply_kind(retriever):
    replies = [
        ask("I have chest pain", retriever),
        ask("knee replacement recovery", retriever),
        ask("sore throat", retriever, api_key=None),
        ask("sore throat", retriever, llm=fake_llm(llm_service.LLMResult(False, error="temporary"))),
        ask("sore throat", retriever, llm=fake_llm(llm_service.LLMResult(True, GOOD))),
    ]
    assert all(config.DISCLAIMER_SHORT in r.content for r in replies)


def test_clarify_then_yes_escalates(retriever):
    first = ask("I feel dizzy", retriever)
    assert first.kind == "clarify"
    history = [{"role": "user", "content": "I feel dizzy"},
               {"role": "assistant", "content": first.content, "kind": "clarify"}]
    assert ask("yes", retriever, history=history).kind == "emergency"


def test_unsupported_question_does_not_use_model_knowledge(retriever):
    def boom(*a, **k):
        raise AssertionError("LLM must not be called without supporting passages")
    r = ask("knee replacement surgery recovery", retriever, llm=boom)
    assert r.kind == "unsupported" and "do not cover" in r.content


def test_missing_key_still_returns_reference_passages(retriever):
    r = ask("sore throat", retriever, api_key=None)
    assert r.kind == "no_key" and r.sources and "[S1]" in r.content


def test_good_answer_has_cited_sources(retriever):
    r = ask("sore throat for two days", retriever, llm=fake_llm(llm_service.LLMResult(True, GOOD)))
    assert r.kind == "answer"
    assert r.sources[0]["publisher"] and r.sources[0]["url"] and r.sources[0]["version"]


def test_blank_model_output_falls_back_to_passages(retriever):
    llm = fake_llm(llm_service.LLMResult(False, error="empty"))
    r = ask("sore throat", retriever, llm=llm)
    assert r.kind == "error" and "temporarily unavailable" in r.content and r.sources


def test_invalid_credentials_message_has_no_raw_details(retriever):
    r = ask("sore throat", retriever, llm=fake_llm(llm_service.LLMResult(False, error="auth")))
    assert "credentials are not valid" in r.content
    assert "INVALID" not in r.content and "400" not in r.content


def test_non_compliant_answer_is_retried_then_accepted(retriever):
    llm = fake_llm(llm_service.LLMResult(True, "just rest"), llm_service.LLMResult(True, GOOD))
    r = ask("sore throat", retriever, llm=llm)
    assert r.kind == "answer" and len(llm.seen) == 2


def test_non_compliant_answer_twice_falls_back(retriever):
    llm = fake_llm(llm_service.LLMResult(True, "just rest"), llm_service.LLMResult(True, "still bad"))
    r = ask("sore throat", retriever, llm=llm)
    assert r.kind == "error" and r.problems


def test_followup_keeps_earlier_symptom_context(retriever):
    history = [
        {"role": "user", "content": "I have a sore throat", "profile": profile().to_dict()},
        {"role": "assistant", "content": GOOD, "kind": "answer"},
    ]
    llm = fake_llm(llm_service.LLMResult(True, GOOD))
    r = ask("It started yesterday", retriever, llm=llm, history=history)
    assert r.kind == "answer"  # retrieval used the earlier message
    sent = " ".join(m["content"] for m in llm.seen[0])
    assert "I have a sore throat" in sent and "It started yesterday" in sent


def test_stored_profile_used_for_earlier_turns(retriever):
    old = profile(age_value=70, severity=8, duration="More than 2 weeks")
    history = [{"role": "user", "content": "I have a sore throat", "profile": old.to_dict()},
               {"role": "assistant", "content": GOOD, "kind": "answer"}]
    llm = fake_llm(llm_service.LLMResult(True, GOOD))
    ask("It started yesterday", retriever, llm=llm, history=history, profile=profile(age_value=25))
    sent = " ".join(m["content"] for m in llm.seen[0])
    assert "70 years" in sent and "25 years" in sent


def test_history_is_bounded(retriever):
    history = []
    for i in range(20):
        history += [{"role": "user", "content": f"question {i}", "profile": profile().to_dict()},
                    {"role": "assistant", "content": GOOD, "kind": "answer"}]
    llm = fake_llm(llm_service.LLMResult(True, GOOD))
    ask("sore throat", retriever, llm=llm, history=history)
    assert len(llm.seen[0]) <= config.MAX_HISTORY_MESSAGES + 2  # system + bounded history + current


def test_infants_under_12_months_are_out_of_scope(retriever):
    r = ask("sore throat", retriever, profile=profile(age_value=3, age_unit="months"))
    assert r.kind == "scope"
    r2 = ask("sore throat", retriever, profile=profile(age_value=2, age_unit="weeks"))
    assert r2.kind == "scope"
    assert ask("sore throat", retriever, profile=profile(age_value=12, age_unit="months"),
               llm=fake_llm(llm_service.LLMResult(True, GOOD))).kind == "answer"


def test_high_severity_note(retriever):
    r = ask("sore throat", retriever, profile=profile(severity=9),
            llm=fake_llm(llm_service.LLMResult(True, GOOD)))
    assert "very severe" in r.content
