from types import SimpleNamespace

import llm_service

GOOD = (
    "**Possible common causes:** A viral infection [S1].\n"
    "**Self-care tips:** Rest and fluids [S1].\n"
    "**See a doctor if:** It lasts over a week [S1].\n"
    "**Emergency signs:** Unable to swallow [S1].\n"
    "Please consult a doctor for proper advice."
)


class FakeClient:
    def __init__(self, outcomes):
        self.outcomes = list(outcomes)
        self.calls = 0
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls += 1
        outcome = self.outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))])


class AuthenticationError(Exception):
    status_code = 401


def run(outcomes, attempts=3):
    client = FakeClient(outcomes)
    result = llm_service.generate([], api_key="k", model="m", attempts=attempts,
                                  sleep=lambda s: None, client_factory=lambda key: client)
    return result, client


def test_missing_key_never_builds_a_client():
    def boom(key):
        raise AssertionError("client must not be created")
    r = llm_service.generate([], api_key=None, model="m", client_factory=boom)
    assert (r.ok, r.error) == (False, "missing_key")


def test_success():
    r, _ = run([GOOD])
    assert r.ok and r.text == GOOD


def test_whitespace_only_output_is_treated_as_empty():
    r, client = run(["   \n ", "", None])
    assert (r.ok, r.error) == (False, "empty") and client.calls == 3


def test_temporary_failure_retries_then_recovers():
    r, client = run([RuntimeError("boom"), GOOD])
    assert r.ok and client.calls == 2


def test_temporary_failure_gives_up():
    r, client = run([RuntimeError("x")] * 3)
    assert (r.ok, r.error) == (False, "temporary") and client.calls == 3


def test_invalid_credentials_are_not_retried():
    r, client = run([AuthenticationError("bad key"), GOOD])
    assert (r.ok, r.error) == (False, "auth") and client.calls == 1


def test_raw_provider_details_not_in_result():
    r, _ = run([RuntimeError("secret provider detail")] * 3)
    assert "secret" not in r.text and "secret" not in r.error


def test_validate_accepts_good_answer():
    assert llm_service.validate_answer(GOOD, 1) == []


def test_validate_blank_and_missing_parts():
    assert llm_service.validate_answer("  \n", 1) == ["answer is blank"]
    problems = llm_service.validate_answer("**Possible common causes:** x [S1]", 1)
    assert any("missing section" in p for p in problems)
    assert "missing closing limitation" in problems


def test_validate_requires_real_citations():
    no_cite = GOOD.replace("[S1]", "")
    assert "no citations" in llm_service.validate_answer(no_cite, 1)
    assert any("does not exist" in p for p in llm_service.validate_answer(GOOD.replace("[S1]", "[S5]"), 1))


def test_validate_flags_dosage():
    assert any("dose" in p for p in llm_service.validate_answer(GOOD + " Take 500 mg.", 1))
