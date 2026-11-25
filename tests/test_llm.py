"""Gemini adapter tests with a FAKE client (no network, no key): checks that the adapter sends the
tool declarations and turns a real google-genai response object into an Action."""
from types import SimpleNamespace

from google.genai import types

import pytest

from src.llm import GeminiExtractor, GeminiLLM, MockLLM, make_llm
from src.state import AgentState
from src.tools import PermanentError, TransientError, run_tool, use_extractor


class FakeModels:
    """MOCK of client.models: records the request and returns a prepared response."""

    def __init__(self, response):
        self.response, self.requests = response, []

    def generate_content(self, model, contents, config):
        self.requests.append({"model": model, "contents": contents, "config": config})
        return self.response


def fake_client(response):
    return SimpleNamespace(models=FakeModels(response))


def response_with_parts(parts):
    return types.GenerateContentResponse(candidates=[types.Candidate(content=types.Content(role="model", parts=parts))])


def test_gemini_function_call_becomes_tool_action():
    resp = response_with_parts([types.Part(function_call=types.FunctionCall(
        name="search_question_bank", args={"skill": "docker", "top_k": 2}))])
    client = fake_client(resp)
    llm = GeminiLLM(client=client, model="test-model")
    action = llm.next_action(AgentState(thread_id="t", user_id="u"), ["system", "user: hi"])
    assert action.kind == "tool" and action.tool == "search_question_bank" and action.args["skill"] == "docker"
    sent = client.models.requests[0]
    declared = {f.name for f in sent["config"].tools[0].function_declarations}
    assert declared == {"parse_resume", "parse_vacancy", "compute_gap", "search_question_bank", "save_prep_plan"}
    assert '"questions_collected": 0' in sent["contents"]


def test_gemini_text_reply_becomes_final_action():
    llm = GeminiLLM(client=fake_client(response_with_parts([types.Part(text="Plan is ready.")])), model="m")
    action = llm.next_action(AgentState(thread_id="t", user_id="u"), ["system"])
    assert action.kind == "final" and action.text == "Plan is ready."


def test_make_llm_defaults_to_mock_without_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    assert isinstance(make_llm(), MockLLM)


def test_mock_answer_uses_only_the_context():
    llm = MockLLM()
    assert "junior devops" in llm.answer(["sys", "[note] Target role: Junior DevOps"]).lower()
    assert "don't know" in llm.answer(["sys", "user: Practice question on bash."])


# ---------------------------------------------------------------- extraction by Gemini (fake client)
class FailingModels:
    def __init__(self, code):
        self.code = code

    def generate_content(self, model, contents, config):
        err = RuntimeError("MOCK API error")
        err.code = self.code
        raise err


def test_gemini_extractor_returns_json_and_tool_validates_it():
    text = '{"skills": [{"skill": "python", "level": "junior", "years": 1.5, "source_line": "1.5 years Python, Linux (Ubuntu), Git."}]}'
    client = fake_client(response_with_parts([types.Part(text=text)]))
    ext = GeminiExtractor(client=client, model="m")
    with use_extractor(ext):
        res, rec = run_tool("parse_resume", {"resume_text": "Junior Engineer: 1.5 years Python, Linux (Ubuntu), Git."})
    assert rec["status"] == "ok" and [s["skill"] for s in res["skills"]] == ["python"]
    cfg = client.models.requests[0]["config"]
    assert cfg.response_mime_type == "application/json" and "skills" in cfg.response_json_schema["properties"]


def test_gemini_extractor_invalid_json_is_permanent():
    ext = GeminiExtractor(client=fake_client(response_with_parts([types.Part(text="not json")])), model="m")
    with pytest.raises(PermanentError):
        ext.extract_resume("Python developer")


@pytest.mark.parametrize("code,error", [(503, TransientError), (429, TransientError), (400, PermanentError)])
def test_gemini_extractor_error_classes(code, error, monkeypatch):
    monkeypatch.setattr("src.llm.time.sleep", lambda s: None)  # no real waiting between retries in tests
    ext = GeminiExtractor(client=SimpleNamespace(models=FailingModels(code)), model="m")
    with pytest.raises(error):
        ext.extract_vacancy("Junior DevOps. Mandatory: Docker.")


def test_gemini_llm_brings_its_extractor():
    llm = GeminiLLM(client=fake_client(response_with_parts([types.Part(text="ok")])), model="m")
    assert isinstance(llm.extractor, GeminiExtractor)


# ---------------------------------------------------------------- choosing an available Gemini model
from src.llm import DEFAULT_GEMINI_MODEL, pick_gemini_model


def listing(*names, action="generateContent"):
    models = [SimpleNamespace(name=f"models/{n}", supported_actions=[action]) for n in names]
    return SimpleNamespace(models=SimpleNamespace(list=lambda: models))


def test_pick_model_keeps_the_preferred_one_when_available():
    assert pick_gemini_model(listing("gemini-3.6-flash", DEFAULT_GEMINI_MODEL)) == DEFAULT_GEMINI_MODEL


def test_pick_model_falls_back_to_newest_flash_when_preferred_is_retired():
    client = listing("gemini-3.6-flash", "gemini-3.8-flash", "gemini-3.8-flash-lite", "gemini-3.8-pro")
    assert pick_gemini_model(client) == "gemini-3.8-flash"


def test_pick_model_when_listing_fails():
    broken = SimpleNamespace(models=SimpleNamespace(list=lambda: (_ for _ in ()).throw(RuntimeError("no network"))))
    assert pick_gemini_model(broken) == DEFAULT_GEMINI_MODEL


# ---------------------------------------------------------------- retries on temporary API errors
from src.llm import generate_with_retry


class Err(Exception):
    def __init__(self, code, msg="MOCK API error"):
        super().__init__(msg)
        self.code = code


class SometimesFailing:
    """MOCK of client.models: fails with the given errors first, then answers."""
    def __init__(self, errors):
        self.errors, self.calls = list(errors), 0

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return "ok"


def test_rate_limit_is_retried_with_the_wait_the_api_asks_for():
    models, waits = SometimesFailing([Err(429, "Quota exceeded. Please retry in 7.5s."), Err(503)]), []
    client = SimpleNamespace(models=models)
    assert generate_with_retry(client, "m", "hi", None, sleep=waits.append) == "ok"
    assert models.calls == 3 and waits == [8.5, 10.0]


def test_daily_quota_and_bad_request_are_not_retried():
    for err in (Err(429, "Quota exceeded for metric ...PerDay..."), Err(400)):
        models = SometimesFailing([err])
        with pytest.raises(Err):
            generate_with_retry(SimpleNamespace(models=models), "m", "hi", None, sleep=lambda s: None)
        assert models.calls == 1
