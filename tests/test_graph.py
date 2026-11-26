"""LangGraph tests (milestone 2). MOCK model only: no API key, no cost, no network.

The four required routes (one test group each):
  1. existing data      -> success
  2. missing information -> interrupt() -> Command(resume=...) -> success
  3. transient failure  -> retry -> success (or a controlled end when the retries are used up)
  4. step budget used up -> controlled stop
plus: checkpointer with thread_id (stop/resume, two separate conversations), the approval gate,
reducers, long-term memory, equivalence with the lesson 4 loop, and the diagram/traces in docs/.
"""
import json
from pathlib import Path

import pytest
from langgraph.types import Command

import src.tools as tools
from src import graph as g
from src.agent import SAMPLE_RESUME, SAMPLE_VACANCY, run_interview_agent
from src.llm import Action, MockLLM, ScriptedLLM
from src.state import NoteStore, ThreadStore

ROOT = Path(__file__).resolve().parent.parent
APPROVED_TOOLS = {"parse_resume", "parse_vacancy", "compute_gap", "search_question_bank", "save_prep_plan"}


@pytest.fixture(autouse=True)
def plans_root(tmp_path, monkeypatch):
    monkeypatch.setattr(tools, "PLANS_ROOT", tmp_path)
    return tmp_path


def node_sequence(trace):
    return [e["node"] for e in trace if e["phase"] == "node"]


def tool_calls(trace):
    return [e for e in trace if e["phase"] == "tool_call"]


# ------------------------------------------------------------------------------
# Route 1: existing data -> success
# ------------------------------------------------------------------------------

def test_route1_existing_data_success(plans_root):
    trace, snap = g.run_route_existing_data()
    v = snap.values
    assert snap.next == () and v["status"] == "DONE"
    assert v["gap_json"] == {"matched": ["python", "linux"], "missing": ["docker", "ci/cd"], "weak": []}
    assert len(v["candidate_questions"]) == 10 and all(q["source_line"] for q in v["candidate_questions"])
    assert v["step"] == 10 and v["max_steps"] == 14
    assert list(plans_root.glob("*.json"))
    # the graph integrates ALL five tools of the approved proposal
    assert {r["tool"] for r in v["tool_log"]} == APPROVED_TOOLS == set(tools.TOOLS)
    # model -> tool loop: every tool call is a decide -> act pair, then the model's final decision
    seq = node_sequence(trace)
    assert seq[:3] == ["intake", "decide", "act"] and seq[-2:] == ["decide", "finish"]
    assert seq.count("act") == 9 and seq.count("decide") == 10 and "approval_gate" in seq
    # raw resume / vacancy text is not kept after the run
    assert v["resume_text"] == "" and v["vacancy_text"] == ""


def test_graph_matches_the_lesson4_loop(plans_root):
    """The conversion must not change behaviour: same gap, same questions, same tool order, same steps."""
    base = run_interview_agent("daria", "t-1", SAMPLE_RESUME, SAMPLE_VACANCY, ThreadStore(), NoteStore(),
                               llm=MockLLM(), base_delay=0.01)
    _, snap = g.run_route_existing_data()
    v = snap.values
    assert v["status"] == base["status"]
    assert v["gap_json"] == base["gap"]
    assert [q["doc_id"] for q in v["candidate_questions"]] == [q["doc_id"] for q in base["questions"]]
    assert [r["tool"] for r in v["tool_log"]] == [r["tool"] for r in base["state"].tool_log]
    assert v["step"] == base["state"].step


# ------------------------------------------------------------------------------
# Route 2: missing information -> interrupt -> resume
# ------------------------------------------------------------------------------

def test_route2_missing_info_stops_at_interrupt_and_resumes_with_a_new_graph_object(plans_root):
    saver = g.InMemorySaver()
    trace = []
    first = g.build_interview_graph(checkpointer=saver)
    snap = g.stream_run(first, g.initial_state("daria", "t-miss", SAMPLE_RESUME, ""), "t-miss", trace)
    # the run really stopped inside the clarify node and called no tool
    assert snap.next == ("clarify",)
    paused = [e for e in trace if e["phase"] == "paused_at_interrupt"][0]["interrupt"]
    assert paused["reason"] == "missing_input" and paused["missing"] == ["vacancy_text"]
    assert not tool_calls(trace) and snap.values.get("saved_plan_path") is None
    assert not list(plans_root.glob("*.json"))

    # the state lives in the checkpointer, so a NEW graph object can resume the same thread
    second = g.build_interview_graph(checkpointer=saver)
    snap = g.stream_run(second, Command(resume=SAMPLE_VACANCY.strip()), "t-miss", trace)
    v = snap.values
    assert snap.next == () and v["status"] == "DONE"
    assert v["gap_json"]["missing"] == ["docker", "ci/cd"] and len(v["candidate_questions"]) == 10
    assert node_sequence(trace).count("clarify") == 1
    assert [e for e in trace if e["phase"] == "clarification_received"][0]["fields_updated"] == ["vacancy_text"]


def test_route2_dict_reply_can_fill_both_fields(plans_root):
    graph = g.build_interview_graph()
    trace = []
    g.stream_run(graph, g.initial_state("daria", "t-both", "", ""), "t-both", trace)
    snap = g.stream_run(graph, Command(resume={"resume_text": SAMPLE_RESUME, "vacancy_text": SAMPLE_VACANCY}),
                        "t-both", trace)
    assert snap.values["status"] == "DONE"


def test_unanswered_clarification_ends_in_a_controlled_way(plans_root):
    graph = g.build_interview_graph()
    trace = []
    snap = g.stream_run(graph, g.initial_state("daria", "t-bad", SAMPLE_RESUME, ""), "t-bad", trace)
    assert snap.next == ("clarify",)
    snap = g.stream_run(graph, Command(resume="no"), "t-bad", trace)  # too short: asked again
    assert snap.next == ("clarify",)
    snap = g.stream_run(graph, Command(resume="no"), "t-bad", trace)  # second failure: controlled end
    assert snap.next == () and snap.values["status"] == "clarification_needed"
    assert not tool_calls(trace) and not list(plans_root.glob("*.json"))


# ------------------------------------------------------------------------------
# Route 3: transient failure -> retry -> success / controlled end
# ------------------------------------------------------------------------------

def test_route3_transient_failure_is_retried_and_then_succeeds(plans_root):
    trace, snap = g.run_route_transient_failure(failures=1)
    v = snap.values
    assert v["status"] == "DONE" and list(plans_root.glob("*.json"))
    retried = [c for c in tool_calls(trace) if c["attempts"] == 2]
    assert len(retried) == 1 and retried[0]["tool"] == "search_question_bank" and retried[0]["status"] == "ok"
    assert "MOCK" in retried[0]["retries"][0]  # the failed attempt is visible in the trace
    assert v["retry_count"] == 1 and v["step"] == 11  # the retry cost one step of the budget


def test_route3_persistent_failure_ends_in_a_controlled_way(plans_root):
    trace, snap = g.run_route_transient_failure(failures=2)
    v = snap.values
    assert v["status"] == "tool_failed" and snap.next == ()
    assert v["saved_plan_path"] is None and not list(plans_root.glob("*.json"))
    failed = [c for c in tool_calls(trace) if c["status"] == "error"][0]
    assert failed["error_type"] == "transient" and failed["attempts"] == 2
    assert "unavailable" in v["answer"]


# ------------------------------------------------------------------------------
# Route 4: step budget used up
# ------------------------------------------------------------------------------

def test_route4_step_budget_exhaustion_stops_the_run(plans_root):
    trace, snap = g.run_route_step_budget(max_steps=3)
    v = snap.values
    assert v["status"] == "stopped_step_budget" and v["step"] == 3 and snap.next == ()
    assert len(v["tool_log"]) == 3 and v["saved_plan_path"] is None and not list(plans_root.glob("*.json"))
    assert node_sequence(trace)[-3:] == ["decide", "budget_stop", "finish"]
    assert [e for e in trace if e["phase"] == "budget_stop"][0]["max_steps"] == 3
    assert "step budget of 3" in v["answer"]


def test_a_retry_never_exceeds_the_remaining_budget(plans_root):
    """A retry costs a step, so the number of attempts is limited by what is left of the budget."""
    def run(max_steps):
        trace = []
        with g.flaky_tool("search_question_bank", failures=1):
            snap = g.stream_run(g.build_interview_graph(),
                                g.initial_state("daria", f"t-tight-{max_steps}", SAMPLE_RESUME, SAMPLE_VACANCY,
                                                max_steps=max_steps), f"t-tight-{max_steps}", trace)
        return snap.values, [c for c in tool_calls(trace) if c["tool"] == "search_question_bank"][0]

    v, call = run(5)  # 2 steps left for the first search: one retry fits, then the budget is used up
    assert call["attempts"] == 2 and call["status"] == "ok"
    assert v["step"] == 5 and v["status"] == "stopped_step_budget"
    v, call = run(4)  # only 1 step left: no retry is possible, the failure ends the run in a controlled way
    assert call["attempts"] == 1 and call["error_type"] == "transient"
    assert v["step"] == 4 and v["status"] == "tool_failed"


# ------------------------------------------------------------------------------
# Checkpointer and thread_id
# ------------------------------------------------------------------------------

def test_two_threads_on_one_checkpointer_do_not_mix(plans_root):
    trace, snaps = g.run_checkpointer_threads_demo()
    a, b = snaps["A"].values, snaps["B"].values
    iso = [e for e in trace if e["phase"] == "isolation_check"][0]
    # while A was stopped, B ran to the end and A was untouched
    assert iso["thread_A_still_waiting_at"] == ["clarify"] and iso["thread_A_gap"] is None
    assert iso["thread_B_status"] == "DONE"
    # after A was resumed the two conversations have their own results
    assert a["user_id"] == "daria" and b["user_id"] == "alina"
    assert a["gap_json"]["missing"] == ["docker", "ci/cd"] and a["gap_json"] != b["gap_json"]
    assert {q["doc_id"] for q in a["candidate_questions"]} != {q["doc_id"] for q in b["candidate_questions"]}
    # every thread has its own tool log: steps are consecutive, nothing from the other thread is mixed in
    assert [r["step"] for r in a["tool_log"]] == list(range(1, len(a["tool_log"]) + 1))
    assert [r["step"] for r in b["tool_log"]] == list(range(1, len(b["tool_log"]) + 1))


def test_a_thread_belongs_to_its_user(plans_root):
    graph = g.build_interview_graph()
    g.stream_run(graph, g.initial_state("daria", "t-own", SAMPLE_RESUME, SAMPLE_VACANCY), "t-own", [])
    with pytest.raises(PermissionError):
        g.stream_run(graph, g.initial_state("alina", "t-own", SAMPLE_RESUME, SAMPLE_VACANCY), "t-own", [])


def test_the_checkpointer_keeps_a_history_of_every_thread(plans_root):
    graph = g.build_interview_graph()
    g.stream_run(graph, g.initial_state("daria", "t-hist", SAMPLE_RESUME, SAMPLE_VACANCY), "t-hist", [])
    history = list(graph.get_state_history(g.thread_config("t-hist")))
    assert len(history) > 10 and history[0].values["status"] == "DONE"


# ------------------------------------------------------------------------------
# Approval gate: the USER decides, the CODE enforces
# ------------------------------------------------------------------------------

def test_approval_gate_blocks_save_without_user_approval(plans_root):
    graph = g.build_interview_graph()
    trace = []
    snap = g.stream_run(graph, g.initial_state("daria", "t-no", SAMPLE_RESUME, SAMPLE_VACANCY,
                                               user_approval=False), "t-no", trace)
    assert snap.values["saved_plan_path"] is None and not list(plans_root.glob("*.json"))
    save = [c for c in tool_calls(trace) if c["tool"] == "save_prep_plan"][0]
    assert save["decision"] == "rejected" and save["error_type"] == "permission_denied"


def test_approval_can_be_a_real_interrupt(plans_root):
    graph = g.build_interview_graph()
    trace = []
    snap = g.stream_run(graph, g.initial_state("daria", "t-ask", SAMPLE_RESUME, SAMPLE_VACANCY,
                                               user_approval=None), "t-ask", trace)
    assert snap.next == ("approval_gate",)
    asked = [e for e in trace if e["phase"] == "paused_at_interrupt"][0]["interrupt"]
    assert asked["reason"] == "approval_required" and asked["question_count"] == 10
    assert not list(plans_root.glob("*.json"))
    snap = g.stream_run(graph, Command(resume=True), "t-ask", trace)
    assert snap.values["saved_plan_path"] and list(plans_root.glob("*.json"))


def test_a_refusal_at_the_interrupt_saves_nothing(plans_root):
    graph = g.build_interview_graph()
    trace = []
    g.stream_run(graph, g.initial_state("daria", "t-ref", SAMPLE_RESUME, SAMPLE_VACANCY, user_approval=None),
                 "t-ref", trace)
    snap = g.stream_run(graph, Command(resume=False), "t-ref", trace)
    assert snap.values["saved_plan_path"] is None and not list(plans_root.glob("*.json"))


def test_a_truthy_string_is_not_an_approval(plans_root):
    graph = g.build_interview_graph()
    trace = []
    g.stream_run(graph, g.initial_state("daria", "t-str", SAMPLE_RESUME, SAMPLE_VACANCY, user_approval=None),
                 "t-str", trace)
    snap = g.stream_run(graph, Command(resume="no"), "t-str", trace)
    assert snap.values["approved"] is False and not list(plans_root.glob("*.json"))


def test_model_cannot_approve_by_itself(plans_root):
    llm = ScriptedLLM([Action("tool", "save_prep_plan", {"plan_name": "x_plan", "approved": True})])
    graph = g.build_interview_graph(llm=llm)
    snap = g.stream_run(graph, g.initial_state("daria", "t-forge", SAMPLE_RESUME, SAMPLE_VACANCY,
                                               user_approval=False), "t-forge", [])
    assert snap.values["saved_plan_path"] is None and not list(plans_root.glob("*.json"))


# ------------------------------------------------------------------------------
# Reducers and memory
# ------------------------------------------------------------------------------

def test_reducer_merge_questions_dedupes_and_caps():
    def q(i):
        return {"doc_id": f"q{i}"}

    assert g.merge_questions([q(1)], [q(1), q(2)]) == [q(1), q(2)]
    assert len(g.merge_questions([], [q(i) for i in range(30)])) == g.MAX_QUESTIONS


def test_accumulating_fields_keep_everything_in_order(plans_root):
    _, snap = g.run_route_existing_data()
    v = snap.values
    assert [r["step"] for r in v["tool_log"]] == list(range(1, 10))
    assert v["searched_skills"] == ["docker", "ci/cd", "python", "linux", "teamwork"]
    phases = [e["phase"] for e in v["trace_events"]]
    assert phases[0] == "input" and phases[-1] == "done"


def test_long_term_memory_is_written_only_after_an_approved_save(plans_root):
    notes = NoteStore()
    graph = g.build_interview_graph(notes=notes)
    g.stream_run(graph, g.initial_state("daria", "t-mem1", SAMPLE_RESUME, SAMPLE_VACANCY, user_approval=False),
                 "t-mem1", [])
    assert notes.all("daria") == []
    g.stream_run(graph, g.initial_state("daria", "t-mem2", SAMPLE_RESUME, SAMPLE_VACANCY, user_approval=True),
                 "t-mem2", [])
    assert any("Junior DevOps" in n for n in notes.all("daria")) and notes.all("alina") == []


# ------------------------------------------------------------------------------
# Documentation artefacts must be what the code produces
# ------------------------------------------------------------------------------

def test_the_mermaid_diagram_in_docs_is_the_compiled_graph():
    expected = g.mermaid_diagram().strip()
    assert (ROOT / "docs" / "graph.mmd").read_text(encoding="utf-8").strip() == expected
    assert expected in (ROOT / "docs" / "milestone2.md").read_text(encoding="utf-8")
    for node in ("intake", "clarify", "decide", "approval_gate", "act", "budget_stop", "finish"):
        assert node in expected


def test_committed_graph_traces_are_what_the_code_produces(tmp_path):
    g.generate_graph_traces(str(tmp_path / "traces"), diagram_path=None)

    def normalise(row):
        """latency is a measurement, and the folder of the saved plan differs (tmp folder in tests)."""
        row = {k: v for k, v in row.items() if k != "latency_ms"}
        if isinstance(row.get("observation"), dict) and "file_path" in row["observation"]:
            row["observation"] = {**row["observation"], "file_path": Path(row["observation"]["file_path"]).name}
        if row.get("saved_plan_path"):
            row["saved_plan_path"] = Path(row["saved_plan_path"]).name
        return row

    def load(path):
        return [normalise(json.loads(line)) for line in path.read_text(encoding="utf-8").splitlines()]

    names = ["route1_existing_data", "route2_missing_info_interrupt", "route3_transient_retry",
             "route4_step_budget", "checkpointer_two_threads"]
    for name in names:
        committed = ROOT / "docs" / "traces" / f"trace_graph_{name}.jsonl"
        regenerated = tmp_path / "traces" / f"trace_graph_{name}.jsonl"
        assert committed.exists(), f"missing trace file {committed.name}"
        assert load(committed) == load(regenerated), f"{committed.name} is not what the code produces: re-run python -m src.graph"
