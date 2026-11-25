"""State & memory tests: pure transitions, step budget, thread/user isolation, memory safety,
and the full-history vs reduced-context comparison (with the MOCK model)."""
import pytest

from src.llm import MockLLM
from src.state import (
    AgentState,
    NoteStore,
    ThreadStore,
    apply_event,
    build_context,
    compare_context_strategies,
    context_tokens,
    run_memory_tool,
)


def test_apply_event_is_pure():
    s0 = AgentState(thread_id="t1", user_id="daria")
    s1 = apply_event(s0, "user", "hello")
    assert len(s0.messages) == 0 and len(s1.messages) == 1


def test_each_tool_event_is_one_step_and_budget_is_enforced():
    s = AgentState(thread_id="t1", user_id="daria", max_steps=2)
    s = apply_event(s, "user", "question")          # user messages do not use the budget
    s = apply_event(s, "tool", "obs 1")
    s = apply_event(s, "tool", "obs 2")
    assert s.step == 2
    with pytest.raises(RuntimeError):
        apply_event(s, "tool", "obs 3")


def test_state_schema_rejects_unknown_fields():
    with pytest.raises(Exception):
        AgentState(thread_id="t1", user_id="daria", is_admin=True)


def test_thread_isolation_between_users():
    ts = ThreadStore()
    s_daria = ts.get_or_create("thread-1", "daria")
    ts.save(apply_event(s_daria, "user", "Daria private resume details"))

    with pytest.raises(PermissionError):
        ts.get_or_create("thread-1", "alina")

    s_alina = ts.get_or_create("thread-2", "alina")
    assert s_alina.messages == []


def test_memory_isolation_and_forged_user_id():
    notes = NoteStore()
    ok_res = run_memory_tool(notes, "daria", "save_note", {"text": "Target role: Junior DevOps"})
    assert ok_res["status"] == "ok"
    assert notes.search("alina", "Junior DevOps") == []

    forged = run_memory_tool(notes, "daria", "save_note", {"text": "Alina note", "user_id": "alina"})
    assert forged["status"] == "error" and forged["error"]["type"] == "invalid_input"
    assert notes.all("alina") == []


def test_notes_are_not_duplicated():
    notes = NoteStore()
    for _ in range(3):
        run_memory_tool(notes, "daria", "save_note", {"text": "Missing skills: docker"})
    assert notes.all("daria") == ["Missing skills: docker"]


def test_sensitive_data_blocked_in_memory():
    notes = NoteStore()
    for bad in ("my password is 12345", "email daria@example.com", "id 123456789"):
        res = run_memory_tool(notes, "daria", "save_note", {"text": bad})
        assert res["status"] == "error" and res["error"]["type"] == "sensitive_data"


def test_context_full_vs_compact_sizes():
    s = AgentState(thread_id="t-ctx", user_id="daria")
    s = apply_event(s, "user", "Preparing for Junior DevOps. I know Python and Linux, missing Docker and CI/CD.")
    s = apply_event(s, "assistant", "Noted: missing Docker and CI/CD.")
    for i in range(6):
        s = apply_event(s, "user", f"Extra question {i} about Linux permissions and systemd.")
        s = apply_event(s, "assistant", f"Answer {i} about systemctl and journalctl.")
    s = apply_event(s, "user", "Remind me which skills I am missing?")

    notes = NoteStore()
    run_memory_tool(notes, "daria", "save_note", {"text": "Missing skills: docker, ci/cd"})

    full_ctx = build_context(s, strategy="full")
    compact_ctx = build_context(s, notes=notes, strategy="notes", last_n=2, query="missing skills")
    assert context_tokens(compact_ctx) < context_tokens(full_ctx)
    assert "docker, ci/cd" in "\n".join(compact_ctx).lower()


def test_comparison_answers_with_mock_model():
    """The same question, one run per strategy: last_n loses the facts, notes keep them."""
    rows = {r["strategy"].split(" (")[0]: r for r in compare_context_strategies(MockLLM())}
    assert rows["full"]["correct"] is True
    assert rows["last_n"]["correct"] is False and rows["last_n"]["lost_in_context"]
    assert rows["notes + last_n"]["correct"] is True
    assert rows["notes + last_n"]["context_chars"] < rows["full"]["context_chars"]
