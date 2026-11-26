"""LangGraph orchestration of the InterviewPrep Agent (milestone 2).

The SAME logic as the milestone 1 / lesson 4 loop in src/agent.py (the MODEL decides, the CODE
validates, runs the tool and updates the state), converted to a LangGraph StateGraph:

    START -> intake -+-> clarify (interrupt: asks the user, then back to intake)
                     +-> decide -+-> act ---------------+-> decide (loop: model -> tool -> observation)
                     |           +-> approval_gate -> act
                     |           +-> budget_stop -> finish
                     |           +-> finish (model says it is done)
                     +-> finish (clarification impossible)            finish -> END

Everything that matters is reused from milestone 1, not copied: the five tool contracts and run_tool
(validation, permissions, timeout, retries), the model layer (MockLLM / GeminiLLM), the two-layer
question filter and the memory notes. Only the control flow moved into the graph.

Run:  python -m src.graph          -> writes docs/traces/trace_graph_*.jsonl and docs/graph.mmd (MockLLM)
      python -m src.graph --live   -> one conversation through the graph with Gemini (needs GEMINI_API_KEY)
"""
import argparse
import json
import operator
from contextlib import contextmanager, nullcontext
from pathlib import Path
from typing import Annotated, Any, Callable, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt

try:
    from langgraph.checkpoint.memory import InMemorySaver
except ImportError:  # older LangGraph releases call it MemorySaver
    from langgraph.checkpoint.memory import MemorySaver as InMemorySaver

import src.tools as tools_module
from src.agent import (MAX_QUESTIONS, SAMPLE_RESUME, SAMPLE_VACANCY, _final_answer, _verify_and_collect,
                       observe, resolve_args)
from src.llm import MockLLM, make_llm
from src.state import MAX_STEPS, AgentState, Message, NoteStore, build_context, run_memory_tool
from src.tools import TransientError, run_tool, use_extractor

MIN_TEXT_CHARS = 15        # same minimum as the input schemas of parse_resume / parse_vacancy
MAX_CLARIFY_ROUNDS = 2     # after two unanswered clarifications the agent ends in a controlled way
RECURSION_LIMIT = 100      # every tool call is 2-3 graph supersteps; LangGraph's default is too small

STATUS_BY_STOP_REASON = {
    "model_final_answer": "DONE",
    "clarification_needed": "clarification_needed",
    "tool_failed": "tool_failed",
    "step_budget_exhausted": "stopped_step_budget",
}


# ==============================================================================
# STATE (TypedDict) AND REDUCERS
# ==============================================================================

def merge_questions(old: list[dict], new: list[dict]) -> list[dict]:
    """Reducer for candidate_questions: appends new questions, drops duplicates by doc_id and
    never exceeds MAX_QUESTIONS, whatever a node returns."""
    merged = list(old or [])
    seen = {q["doc_id"] for q in merged}
    for question in new or []:
        if question["doc_id"] not in seen and len(merged) < MAX_QUESTIONS:
            seen.add(question["doc_id"])
            merged.append(question)
    return merged


class GraphState(TypedDict, total=False):
    """Explicit state of ONE conversation (one thread_id). Fields marked with a reducer are
    accumulated (a node returns only what is NEW); all other fields are overwritten."""
    # identity
    thread_id: str
    user_id: str
    # raw inputs of this run (cleared by `finish`, never copied to long-term memory)
    resume_text: str
    vacancy_text: str
    # results of the tools
    resume_json: Optional[dict]
    vacancy_json: Optional[dict]
    gap_json: Optional[dict]
    searched_skills: Annotated[list[str], operator.add]
    candidate_questions: Annotated[list[dict], merge_questions]
    # approval: user_approval is the USER's answer; approved is what the code derives from it.
    # The model can never set either of them.
    user_approval: Optional[bool]
    approved: bool
    saved_plan_path: Optional[str]
    output_dir: Optional[str]
    # budget and control
    step: int
    max_steps: int
    retry_count: int
    clarify_rounds: int
    next_action: Optional[dict]
    stop_reason: str
    model_text: str
    status: str
    answer: str
    # logs that only grow
    tool_log: Annotated[list[dict], operator.add]
    messages: Annotated[list[dict], operator.add]
    trace_events: Annotated[list[dict], operator.add]


def initial_state(user_id: str, thread_id: str, resume_text: str, vacancy_text: str,
                  max_steps: int = MAX_STEPS, user_approval: Optional[bool] = True,
                  output_dir: Optional[str] = None) -> dict[str, Any]:
    """Input of one run. user_approval=True/False simulates the user's answer at the approval gate;
    None makes the graph pause at the gate (interrupt) and wait for Command(resume=True/False)."""
    return {"user_id": user_id, "thread_id": thread_id, "resume_text": resume_text,
            "vacancy_text": vacancy_text, "max_steps": max_steps, "user_approval": user_approval,
            "output_dir": output_dir,
            # every field of the state has a defined starting value
            "resume_json": None, "vacancy_json": None, "gap_json": None, "saved_plan_path": None,
            "approved": False, "step": 0, "retry_count": 0, "clarify_rounds": 0,
            "next_action": None, "stop_reason": "", "model_text": "", "status": "", "answer": ""}


def thread_config(thread_id: str) -> dict[str, Any]:
    """The checkpointer stores one independent state per thread_id."""
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": RECURSION_LIMIT}


def _missing_inputs(state: GraphState) -> list[str]:
    missing = []
    if len((state.get("resume_text") or "").strip()) < MIN_TEXT_CHARS:
        missing.append("resume_text")
    if len((state.get("vacancy_text") or "").strip()) < MIN_TEXT_CHARS:
        missing.append("vacancy_text")
    return missing


def _agent_state(state: GraphState) -> AgentState:
    """The milestone 1 helpers (model layer, context builder, filter) work on AgentState: this is a
    read-only view of the graph state, so the graph reuses them without copying them."""
    return AgentState(
        thread_id=state.get("thread_id", "thread"), user_id=state.get("user_id", "user"),
        step=state.get("step", 0), max_steps=state.get("max_steps", MAX_STEPS),
        retry_count=state.get("retry_count", 0),
        resume_text=state.get("resume_text", ""), vacancy_text=state.get("vacancy_text", ""),
        resume_json=state.get("resume_json"), vacancy_json=state.get("vacancy_json"),
        gap_json=state.get("gap_json"), searched_skills=list(state.get("searched_skills", [])),
        candidate_questions=list(state.get("candidate_questions", [])),
        approved=bool(state.get("approved", False)), saved_plan_path=state.get("saved_plan_path"),
        tool_log=list(state.get("tool_log", [])),
        messages=[Message(**m) for m in state.get("messages", [])],
    )


# ==============================================================================
# THE GRAPH
# ==============================================================================

def build_interview_graph(llm: Any = None, notes: Optional[NoteStore] = None, checkpointer: Any = None,
                          max_attempts: int = 2, base_delay: float = 0.05,
                          timeout_s: Optional[float] = None):
    """Builds and compiles the graph. llm: MockLLM (default) or GeminiLLM. checkpointer: where the
    per-thread state is saved (InMemorySaver by default)."""
    llm = llm or MockLLM()
    notes = notes if notes is not None else NoteStore()

    # --- node: intake -----------------------------------------------------------
    def intake(state: GraphState) -> dict:
        first_time = not state.get("messages")
        event = {"phase": "input" if first_time else "inputs_rechecked",
                 "thread_id": state.get("thread_id"), "user_id": state.get("user_id"), "model": llm.name,
                 "max_steps": state.get("max_steps", MAX_STEPS),
                 "resume_chars": len(state.get("resume_text") or ""),
                 "vacancy_chars": len(state.get("vacancy_text") or "")}
        update: dict[str, Any] = {"trace_events": [event]}
        if first_time:
            update["messages"] = [{"role": "user", "content": (
                f"Prepare interview questions (resume: {event['resume_chars']} chars, "
                f"vacancy: {event['vacancy_chars']} chars).")}]
        missing = _missing_inputs(state)
        if missing and state.get("clarify_rounds", 0) >= MAX_CLARIFY_ROUNDS:
            update["stop_reason"] = "clarification_needed"
        return update

    def route_after_intake(state: GraphState) -> str:
        if state.get("stop_reason"):
            return "finish"
        return "clarify" if _missing_inputs(state) else "decide"

    # --- node: clarify (real stop for human input) --------------------------------
    def clarify(state: GraphState) -> dict:
        missing = _missing_inputs(state)
        # interrupt() stops the run here; the state stays in the checkpointer under this thread_id.
        # On resume LangGraph re-runs this node and interrupt() returns the value of Command(resume=...).
        reply = interrupt({"reason": "missing_input", "missing": missing,
                           "question": "Some input is missing or too short. Please provide: " + ", ".join(missing)})
        resume_text, vacancy_text = state.get("resume_text", ""), state.get("vacancy_text", "")
        updated = []
        if isinstance(reply, dict):
            if reply.get("resume_text"):
                resume_text, updated = str(reply["resume_text"]), updated + ["resume_text"]
            if reply.get("vacancy_text"):
                vacancy_text, updated = str(reply["vacancy_text"]), updated + ["vacancy_text"]
        else:  # a plain text answer fills the first missing field
            text = str(reply).strip()
            if "resume_text" in missing:
                resume_text, updated = text, ["resume_text"]
            elif "vacancy_text" in missing:
                vacancy_text, updated = text, ["vacancy_text"]
        return {
            "resume_text": resume_text, "vacancy_text": vacancy_text,
            "clarify_rounds": state.get("clarify_rounds", 0) + 1,
            "messages": [{"role": "assistant", "content": "Asked the user for: " + ", ".join(missing)},
                         {"role": "user", "content": f"Clarification received for: {', '.join(updated) or 'nothing'}"}],
            "trace_events": [{"phase": "clarification_received", "asked_for": missing, "fields_updated": updated}],
        }

    # --- node: decide (the MODEL chooses the next tool) ------------------------------
    def decide(state: GraphState) -> dict:
        step, limit = state.get("step", 0), state.get("max_steps", MAX_STEPS)
        if step >= limit:  # the only place that checks the step budget
            return {"stop_reason": "step_budget_exhausted", "next_action": None}
        agent = _agent_state(state)
        ctx = build_context(agent, notes=notes, strategy="notes", last_n=6, query="target role missing skills")
        action = llm.next_action(agent, ctx)
        if action.kind == "final":
            return {"stop_reason": "model_final_answer", "model_text": action.text, "next_action": None,
                    "step": step + 1,  # a model decision is a step
                    "trace_events": [{"phase": "final_decision", "step": step + 1, "decided_by": llm.name}]}
        return {"next_action": {"tool": action.tool or "", "args": dict(action.args)},
                "trace_events": [{"phase": "decision", "step": step + 1, "decided_by": llm.name,
                                  "tool": action.tool, "model_args": dict(action.args),
                                  "context_items": len(ctx)}]}

    def route_after_decide(state: GraphState) -> str:
        reason = state.get("stop_reason", "")
        if reason == "step_budget_exhausted":
            return "budget_stop"
        if reason:
            return "finish"
        if (state.get("next_action") or {}).get("tool") == "save_prep_plan":
            return "approval_gate"
        return "act"

    # --- node: approval_gate (the USER decides, the CODE enforces) ---------------------
    def approval_gate(state: GraphState) -> dict:
        decision = state.get("user_approval")
        if decision is None:  # no answer supplied up front: stop and ask the user
            decision = interrupt({"reason": "approval_required",
                                  "question_count": len(state.get("candidate_questions", [])),
                                  "question": "Save the prepared plan? Reply with Command(resume=True/False)."})
        approved = decision is True  # only a real True counts, never a truthy string
        return {"approved": approved,
                "trace_events": [{"phase": "approval_gate", "approved": approved,
                                  "question_count": len(state.get("candidate_questions", []))}]}

    # --- node: act (validation, permission, timeout, retry are inside run_tool) ------------
    def act(state: GraphState) -> dict:
        action = state["next_action"]
        tool = action["tool"]
        agent = _agent_state(state)
        step, limit = state.get("step", 0), state.get("max_steps", MAX_STEPS)
        args = resolve_args(tool, action["args"], agent, state.get("output_dir"))  # trusted data from the state

        attempts_left = min(max_attempts, limit - step)  # a retry also costs a step
        extractor = getattr(llm, "extractor", None)
        with use_extractor(extractor) if extractor is not None else nullcontext():
            result, rec = run_tool(tool, args, max_attempts=attempts_left, base_delay=base_delay,
                                   timeout_s=timeout_s, step=step + 1)
        obs = observe(tool, result, rec)
        retries = max(0, rec.get("attempts", 1) - 1)
        new_step = step + 1 + retries

        events = [{"phase": "tool_call", "step": new_step, "decided_by": llm.name, "tool": tool,
                   "args": rec["args"], "decision": rec.get("decision"), "status": rec.get("status"),
                   "error_type": rec.get("error_type"), "reason": rec.get("reason", ""),
                   "attempts": rec.get("attempts"), "retries": rec.get("retries", []),
                   "latency_ms": rec.get("latency_ms"), "observation": obs}]
        update: dict[str, Any] = {
            "step": new_step, "retry_count": state.get("retry_count", 0) + retries,
            "tool_log": [rec], "next_action": None,
            "messages": [{"role": "tool", "content": json.dumps(
                {"tool": tool, "status": rec["status"], **obs}, ensure_ascii=False)}],
        }

        if result is not None:
            if tool == "parse_resume":
                update["resume_json"] = result
                if result["injection_warning"]:
                    events.append({"phase": "security_alert",
                                   "detail": "Prompt-injection line in the resume was dropped and treated as data."})
            elif tool == "parse_vacancy":
                update["vacancy_json"] = result
            elif tool == "compute_gap":
                update["gap_json"] = result
            elif tool == "search_question_bank":
                skill = args["skill"]
                update["searched_skills"] = [skill]
                before = len(agent.candidate_questions)
                verified = _verify_and_collect(agent, skill, result["results"], events)
                update["candidate_questions"] = verified.candidate_questions[before:]  # only the NEW ones
            elif tool == "save_prep_plan":
                update["saved_plan_path"] = result["file_path"]
        else:
            # failures the agent must not "work around"
            if tool in ("parse_resume", "parse_vacancy") and rec.get("error_type") in ("invalid_input", "permanent"):
                update["stop_reason"] = "clarification_needed"
            elif rec.get("error_type") == "transient":
                update["stop_reason"] = "tool_failed"
            # permission_denied / invalid model args: the model sees the error and decides again
        update["trace_events"] = events
        return update

    def route_after_act(state: GraphState) -> str:
        return "finish" if state.get("stop_reason") else "decide"

    # --- nodes: budget_stop and finish ----------------------------------------------------
    def budget_stop(state: GraphState) -> dict:
        return {"trace_events": [{"phase": "budget_stop", "step": state.get("step", 0),
                                  "max_steps": state.get("max_steps", MAX_STEPS),
                                  "detail": "step budget used up: the run is stopped in a controlled way"}]}

    def finish(state: GraphState) -> dict:
        agent = _agent_state(state)
        reason = state.get("stop_reason") or "model_final_answer"
        # long-term memory: only an APPROVED, saved plan is remembered, as short facts (never raw text)
        if agent.saved_plan_path and agent.gap_json:
            role = (agent.vacancy_json or {}).get("role", "")
            if role:
                run_memory_tool(notes, agent.user_id, "save_note", {"text": f"Target role: {role}"})
            if agent.gap_json["missing"]:
                run_memory_tool(notes, agent.user_id, "save_note",
                                {"text": "Missing skills: " + ", ".join(agent.gap_json["missing"])})
        answer = _final_answer(agent, reason)
        status = STATUS_BY_STOP_REASON[reason]
        return {
            "answer": answer, "status": status, "stop_reason": reason,
            "resume_text": "", "vacancy_text": "",  # raw text is not kept after the run
            "messages": [{"role": "assistant", "content": answer}],
            "trace_events": [{"phase": "done", "status": status, "stop_reason": reason,
                              "steps_used": agent.step, "max_steps": agent.max_steps,
                              "retries": agent.retry_count, "question_count": len(agent.candidate_questions),
                              "saved": agent.saved_plan_path is not None}],
        }

    builder = StateGraph(GraphState)
    for name, fn in (("intake", intake), ("clarify", clarify), ("decide", decide),
                     ("approval_gate", approval_gate), ("act", act),
                     ("budget_stop", budget_stop), ("finish", finish)):
        builder.add_node(name, fn)
    builder.add_edge(START, "intake")
    builder.add_conditional_edges("intake", route_after_intake,
                                  {"clarify": "clarify", "decide": "decide", "finish": "finish"})
    builder.add_edge("clarify", "intake")
    builder.add_conditional_edges("decide", route_after_decide,
                                  {"act": "act", "approval_gate": "approval_gate",
                                   "budget_stop": "budget_stop", "finish": "finish"})
    builder.add_edge("approval_gate", "act")
    builder.add_conditional_edges("act", route_after_act, {"decide": "decide", "finish": "finish"})
    builder.add_edge("budget_stop", "finish")
    builder.add_edge("finish", END)
    return builder.compile(checkpointer=checkpointer if checkpointer is not None else InMemorySaver())


# ==============================================================================
# RUNNING THE GRAPH AND WRITING TRACES
# ==============================================================================

def stream_run(graph: Any, payload: Any, thread_id: str, trace: list[dict[str, Any]]) -> Any:
    """Runs (or resumes, when payload is a Command) the graph with stream_mode='updates' and appends to
    `trace`: one line per executed node (+ the fields it changed) followed by that node's own events.
    A pause at interrupt() is recorded from the real checkpoint. Returns the state snapshot."""
    config = thread_config(thread_id)
    if isinstance(payload, dict):  # a new run on an existing thread: it must belong to the same user
        owner = graph.get_state(config).values.get("user_id")
        if owner and owner != payload.get("user_id"):
            raise PermissionError("thread belongs to another user")
    for chunk in graph.stream(payload, config, stream_mode="updates"):
        for node, update in chunk.items():
            if node == "__interrupt__":
                for item in update:
                    trace.append({"phase": "paused_at_interrupt", "thread_id": thread_id, "interrupt": item.value})
                continue
            update = update or {}
            trace.append({"phase": "node", "node": node,
                          "changed": sorted(k for k in update if k != "trace_events")})
            trace.extend(update.get("trace_events", []))
    snapshot = graph.get_state(config)
    if snapshot.next:
        trace.append({"phase": "checkpoint", "thread_id": thread_id, "waiting_at": list(snapshot.next),
                      "step": snapshot.values.get("step", 0),
                      "saved_plan_path": snapshot.values.get("saved_plan_path")})
    return snapshot


@contextmanager
def flaky_tool(tool: str, failures: int):
    """MOCK FAILURE (harness for tests and demo traces, not part of the agent): the first `failures`
    calls of `tool` raise a simulated 503; afterwards the real tool runs."""
    original = tools_module.TOOLS[tool]
    calls = {"count": 0}

    def flaky(**kwargs):
        calls["count"] += 1
        if calls["count"] <= failures:
            raise TransientError(f"MOCK: simulated {tool} outage (503)")
        return original(**kwargs)

    tools_module.TOOLS[tool] = flaky
    try:
        yield calls
    finally:
        tools_module.TOOLS[tool] = original


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False, default=str) + "\n")


def mermaid_diagram() -> str:
    return build_interview_graph().get_graph().draw_mermaid()


def run_route_existing_data(llm: Any = None, output_dir: Optional[str] = None) -> tuple[list[dict], Any]:
    """Route 1 (success): resume and vacancy are present."""
    trace: list[dict] = []
    graph = build_interview_graph(llm=llm)
    snap = stream_run(graph, initial_state("daria", "route1-success", SAMPLE_RESUME, SAMPLE_VACANCY,
                                           output_dir=output_dir), "route1-success", trace)
    return trace, snap


def run_route_missing_info(llm: Any = None, output_dir: Optional[str] = None) -> tuple[list[dict], Any]:
    """Route 2: the vacancy is missing -> the graph stops at interrupt() and asks; it is resumed with
    Command(resume=...) through a NEW graph object that shares only the checkpointer."""
    trace: list[dict] = []
    saver = InMemorySaver()
    thread = "route2-interrupt-resume"
    first = build_interview_graph(llm=llm, checkpointer=saver)
    stream_run(first, initial_state("daria", thread, SAMPLE_RESUME, "", output_dir=output_dir), thread, trace)
    trace.append({"phase": "resume", "thread_id": thread, "command": "Command(resume=<vacancy text>)",
                  "resume_chars": len(SAMPLE_VACANCY.strip()), "graph_object": "new graph, same checkpointer"})
    second = build_interview_graph(llm=llm, checkpointer=saver)
    snap = stream_run(second, Command(resume=SAMPLE_VACANCY.strip()), thread, trace)
    return trace, snap


def run_route_transient_failure(llm: Any = None, output_dir: Optional[str] = None,
                                failures: int = 1) -> tuple[list[dict], Any]:
    """Route 3: search_question_bank fails with a simulated 503 `failures` times. One failure -> retry ->
    success; two failures -> the retries are used up and the run ends in a controlled way."""
    trace: list[dict] = []
    graph = build_interview_graph(llm=llm)
    with flaky_tool("search_question_bank", failures):
        snap = stream_run(graph, initial_state("alina", f"route3-transient-{failures}", SAMPLE_RESUME,
                                               SAMPLE_VACANCY, output_dir=output_dir),
                          f"route3-transient-{failures}", trace)
    return trace, snap


def run_route_step_budget(llm: Any = None, output_dir: Optional[str] = None,
                          max_steps: int = 3) -> tuple[list[dict], Any]:
    """Route 4: the step budget (3) is used up after parse_resume, parse_vacancy and compute_gap."""
    trace: list[dict] = []
    graph = build_interview_graph(llm=llm)
    snap = stream_run(graph, initial_state("daria", "route4-budget", SAMPLE_RESUME, SAMPLE_VACANCY,
                                           max_steps=max_steps, output_dir=output_dir), "route4-budget", trace)
    return trace, snap


OTHER_RESUME = ("Data analyst. 3 years experience with Python, pandas and SQL. "
                "Built Docker images for reporting jobs. Basic Git.")
OTHER_VACANCY = "Junior Data Engineer. Requirements: Python, SQL, Docker, Linux, Git."


def run_checkpointer_threads_demo(llm: Any = None, output_dir: Optional[str] = None) -> tuple[list[dict], dict]:
    """ONE checkpointer, TWO conversations: thread A stops at interrupt() (vacancy missing), thread B runs
    to the end meanwhile, then A is resumed. The two states never mix."""
    trace: list[dict] = []
    saver = InMemorySaver()
    graph = build_interview_graph(llm=llm, checkpointer=saver)
    a, b = "demo-thread-A", "demo-thread-B"

    stream_run(graph, initial_state("daria", a, SAMPLE_RESUME, "", output_dir=output_dir), a, trace)
    snap_b = stream_run(graph, initial_state("alina", b, OTHER_RESUME, OTHER_VACANCY, output_dir=output_dir), b, trace)
    snap_a_waiting = graph.get_state(thread_config(a))
    trace.append({"phase": "isolation_check", "thread_A_still_waiting_at": list(snap_a_waiting.next),
                  "thread_A_gap": snap_a_waiting.values.get("gap_json"),
                  "thread_B_status": snap_b.values.get("status"),
                  "thread_B_gap": snap_b.values.get("gap_json")})
    trace.append({"phase": "resume", "thread_id": a, "command": "Command(resume=<vacancy text>)"})
    snap_a = stream_run(graph, Command(resume=SAMPLE_VACANCY.strip()), a, trace)
    history_a = list(graph.get_state_history(thread_config(a)))
    history_b = list(graph.get_state_history(thread_config(b)))
    trace.append({"phase": "threads_summary",
                  "thread_A": {"status": snap_a.values.get("status"), "gap": snap_a.values.get("gap_json"),
                               "tools_called": [r["tool"] for r in snap_a.values.get("tool_log", [])],
                               "checkpoints": len(history_a)},
                  "thread_B": {"status": snap_b.values.get("status"), "gap": snap_b.values.get("gap_json"),
                               "tools_called": [r["tool"] for r in snap_b.values.get("tool_log", [])],
                               "checkpoints": len(history_b)}})
    return trace, {"A": snap_a, "B": snap_b}


def generate_graph_traces(traces_dir: str = "docs/traces", output_dir: Optional[str] = None,
                          diagram_path: Optional[str] = "docs/graph.mmd") -> dict[str, Any]:
    """Runs the four routes (+ the checkpointer demo) with the MockLLM and writes the real trace files."""
    out = Path(traces_dir)
    out.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    for name, runner in (("route1_existing_data", run_route_existing_data),
                         ("route2_missing_info_interrupt", run_route_missing_info),
                         ("route3_transient_retry", run_route_transient_failure),
                         ("route4_step_budget", run_route_step_budget),
                         ("checkpointer_two_threads", run_checkpointer_threads_demo)):
        trace, snap = runner(output_dir=output_dir)
        _write_jsonl(out / f"trace_graph_{name}.jsonl", trace)
        results[name] = {"trace": trace, "snapshot": snap}
    if diagram_path:
        Path(diagram_path).parent.mkdir(parents=True, exist_ok=True)
        Path(diagram_path).write_text(mermaid_diagram() + "\n", encoding="utf-8")
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="InterviewPrep Agent - milestone 2 (LangGraph)")
    parser.add_argument("--live", action="store_true", help="one run through the graph with Gemini (needs GEMINI_API_KEY)")
    args = parser.parse_args()
    if args.live:
        llm = make_llm()
        if llm.name != "gemini":
            raise SystemExit("--live needs GEMINI_API_KEY (environment variable or Colab Secrets)")
        trace: list[dict] = []
        snap = stream_run(build_interview_graph(llm=llm), initial_state("demo-user", "live-01", SAMPLE_RESUME,
                          SAMPLE_VACANCY, user_approval=False), "live-01", trace)
        for entry in trace:
            print(json.dumps(entry, ensure_ascii=False, default=str))
        print("\n" + snap.values["answer"])
        return
    results = generate_graph_traces()
    for name, res in results.items():
        print(f"docs/traces/trace_graph_{name}.jsonl  ({len(res['trace'])} lines)")
    print("docs/graph.mmd  (Mermaid diagram generated from the compiled graph)")


if __name__ == "__main__":
    main()
