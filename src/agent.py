"""Agent loop of InterviewPrep Agent: decide -> act -> observe -> update state -> stop condition.

    while True:
        stop if the step budget is used up
        context = build_context(state)                  # what the model sees
        action  = llm.next_action(state, context)       # the MODEL decides (mock or Gemini)
        if action is final: stop
        args    = trusted data from state + model args  # the CODE supplies trusted data
        result  = run_tool(action.tool, args)           # validation, permission, timeout, retry
        state   = state + observation                   # one step of the budget
        stop on a failure the agent cannot recover from

Run:  python -m src.agent          -> writes docs/traces/*.jsonl with the MockLLM
      python -m src.agent --live   -> runs one conversation with Gemini (needs GEMINI_API_KEY)
"""
import argparse
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

from src.llm import Action, MockLLM, make_llm
from src.state import MAX_STEPS, AgentState, NoteStore, ThreadStore, apply_event, build_context, run_memory_tool
from src.tools import TOOLS, TransientError, check_question_safety, run_tool, use_extractor

# Regular scenario of the proposal (section 5): resume with Python, Linux, Git; a Junior DevOps
# job that requires Docker, CI/CD, Linux and Python -> Docker and CI/CD are the gaps.
SAMPLE_RESUME = """
Junior Software & Systems Engineer.
1.5 years experience with Python scripting for automation, Linux server administration (Ubuntu, systemd), and Git workflows.
Built internal command-line tools for automation.
"""

SAMPLE_VACANCY = """
Junior DevOps Engineer.
Mandatory requirements: Python, Linux, Docker, CI/CD pipelines.
"""

MAX_QUESTIONS = 12
BEHAVIORAL_SKILLS = {"teamwork", "learning"}


def resolve_args(tool: str, model_args: dict[str, Any], state: AgentState, output_dir: str | None) -> dict[str, Any]:
    """Trusted data comes from the state, never from the model. Unknown model args are kept,
    so run_tool rejects them as invalid input (visible in the trace)."""
    if tool == "parse_resume":
        return {"resume_text": state.resume_text}
    if tool == "parse_vacancy":
        return {"vacancy_text": state.vacancy_text}
    if tool == "compute_gap":
        return {"resume_json": state.resume_json or {}, "vacancy_json": state.vacancy_json or {}}
    if tool == "search_question_bank":
        args = {"level": "junior", **model_args}
        return args
    if tool == "save_prep_plan":
        args = {
            "plan_name": str(model_args.get("plan_name") or f"{state.user_id}_{state.thread_id}_plan"),
            "questions": state.candidate_questions,
            "approved": state.approved,  # set by the approval gate, never by the model
        }
        if output_dir:
            args["output_dir"] = output_dir
        return args
    return dict(model_args)


def observe(tool: str, result: dict[str, Any] | None, rec: dict[str, Any]) -> dict[str, Any]:
    """Short observation that is written to the state (and the trace) after each tool call."""
    if result is None:
        return {"error_type": rec.get("error_type"), "reason": rec.get("reason")}
    if tool == "parse_resume":
        return {"skills": [f'{s["skill"]} ({s["years"]}y)' for s in result["skills"]],
                "injection_warning": result["injection_warning"]}
    if tool == "parse_vacancy":
        return {k: result[k] for k in ("role", "must_have", "nice_to_have")}
    if tool == "search_question_bank":
        return {"results": [f'{r["doc_id"]}: {r["topic"]}' for r in result["results"]]}
    return result


def _verify_and_collect(state: AgentState, skill: str, results: list[dict[str, Any]],
                        trace: list[dict[str, Any]]) -> AgentState:
    """Code-side verification of every retrieved question (two-layer filter + dedupe + cap)."""
    gap = state.gap_json or {"matched": [], "missing": [], "weak": []}
    allowed = set(gap["matched"] + gap["missing"] + gap["weak"]) | BEHAVIORAL_SKILLS
    source_map = (state.vacancy_json or {}).get("source_lines", {})
    role = (state.vacancy_json or {}).get("role", "")
    questions = list(state.candidate_questions)
    seen = {q["doc_id"] for q in questions}
    for item in results:
        if item["doc_id"] in seen or len(questions) >= MAX_QUESTIONS:
            continue
        src = source_map.get(skill) or (f"Behavioral baseline for: {role}" if skill in BEHAVIORAL_SKILLS else "")
        ok, reason = check_question_safety(item["question"], allowed, src, context_text=item["topic"])
        if not ok:
            trace.append({"phase": "verification_rejected", "doc_id": item["doc_id"], "reason": reason})
            continue
        seen.add(item["doc_id"])
        questions.append({
            "doc_id": item["doc_id"],
            "skill": skill,
            "topic": item["topic"],
            "category": item["category"],
            "is_gap": skill in gap["missing"] or skill in gap["weak"],
            "source_line": src,
            "question": item["question"],
            "look_for": item["look_for"],
        })
    return state.model_copy(update={"candidate_questions": questions})


def run_interview_agent(
    user_id: str,
    thread_id: str,
    resume_text: str,
    vacancy_text: str,
    threads: ThreadStore,
    notes: NoteStore,
    llm: Any = None,
    approve_save: bool | Callable[[list[dict[str, Any]]], bool] = True,
    max_steps: int = MAX_STEPS,
    output_dir: str | None = None,
    max_attempts: int = 2,
    base_delay: float = 0.05,
    timeout_s: float | None = None,
) -> dict[str, Any]:
    """approve_save simulates the user's answer at the approval gate: a bool, or a callback
    that receives the questions and returns True/False (e.g. input() in a real UI)."""
    llm = llm or MockLLM()
    ask_user = approve_save if callable(approve_save) else (lambda _questions: bool(approve_save))
    trace: list[dict[str, Any]] = []

    state = threads.get_or_create(thread_id, user_id, max_steps=max_steps)
    state = state.model_copy(update={"resume_text": resume_text, "vacancy_text": vacancy_text})
    state = apply_event(state, "user", f"Prepare interview questions (resume: {len(resume_text)} chars, "
                                       f"vacancy: {len(vacancy_text)} chars).")
    trace.append({"phase": "input", "thread_id": thread_id, "user_id": user_id, "model": llm.name,
                  "max_steps": state.max_steps})

    stop_reason = ""
    model_text = ""
    extractor = getattr(llm, "extractor", None)
    with use_extractor(extractor) if extractor is not None else _no_change():
        state, stop_reason, model_text = _loop(state, llm, notes, trace, ask_user, output_dir,
                                               max_attempts, base_delay, timeout_s)

    # --- long-term memory (proposal section 7): only an APPROVED, saved plan is remembered,
    #     as short facts for this user; never the raw resume or vacancy text
    if state.saved_plan_path and state.gap_json:
        role = (state.vacancy_json or {}).get("role", "")
        if role:
            run_memory_tool(notes, user_id, "save_note", {"text": f"Target role: {role}"})
        if state.gap_json["missing"]:
            run_memory_tool(notes, user_id, "save_note",
                            {"text": "Missing skills: " + ", ".join(state.gap_json["missing"])})

    answer = _final_answer(state, stop_reason)
    state = apply_event(state, "assistant", answer)
    # raw text is not kept after the run (proposal section 7)
    state = state.model_copy(update={"resume_text": "", "vacancy_text": ""})
    threads.save(state)
    status = {"model_final_answer": "DONE", "clarification_needed": "clarification_needed",
              "tool_failed": "tool_failed", "step_budget_exhausted": "stopped_step_budget"}[stop_reason]
    trace.append({"phase": "done", "status": status, "stop_reason": stop_reason, "steps_used": state.step,
                  "max_steps": state.max_steps, "retries": state.retry_count,
                  "question_count": len(state.candidate_questions), "saved": state.saved_plan_path is not None})
    return {
        "status": status,
        "answer": answer,
        "model_message": model_text,
        "questions": state.candidate_questions,
        "gap": state.gap_json,
        "saved": {"file_path": state.saved_plan_path} if state.saved_plan_path else None,
        "trace": trace,
        "state": state,
    }


@contextmanager
def _no_change():
    yield


def _loop(state, llm, notes, trace, ask_user, output_dir, max_attempts, base_delay, timeout_s):
    stop_reason, model_text = "", ""
    while True:
        # --- stop condition 1: step budget
        if state.step >= state.max_steps:
            stop_reason = "step_budget_exhausted"
            break

        # --- the MODEL decides the next action from the context built for this thread
        ctx = build_context(state, notes=notes, strategy="notes", last_n=6, query="target role missing skills")
        action: Action = llm.next_action(state, ctx)
        if action.kind == "final":
            stop_reason = "model_final_answer"  # --- stop condition 2: the model says it is done
            model_text = action.text
            state = state.model_copy(update={"step": state.step + 1})  # a model decision is a step
            trace.append({"phase": "final_decision", "step": state.step, "decided_by": llm.name})
            break

        tool = action.tool or ""
        if tool == "save_prep_plan":
            # --- approval gate: the USER decides, the CODE enforces (run_tool rejects approved=False)
            approved = bool(ask_user(state.candidate_questions))
            state = state.model_copy(update={"approved": approved})
            trace.append({"phase": "approval_gate", "approved": approved,
                          "question_count": len(state.candidate_questions)})
        args = resolve_args(tool, action.args, state, output_dir)

        # a retry also costs a step, so attempts never exceed what is left of the budget
        attempts_left = min(max_attempts, state.max_steps - state.step)
        result, rec = run_tool(tool, args, max_attempts=attempts_left, base_delay=base_delay,
                               timeout_s=timeout_s, step=state.step + 1)
        obs = observe(tool, result, rec)
        state = apply_event(state, "tool", json.dumps({"tool": tool, "status": rec["status"], **obs},
                                                      ensure_ascii=False))
        retries = max(0, rec.get("attempts", 1) - 1)
        state = state.model_copy(update={"tool_log": state.tool_log + [rec], "step": state.step + retries,
                                         "retry_count": state.retry_count + retries})
        trace.append({
            "phase": "tool_call", "step": state.step, "decided_by": llm.name, "tool": tool,
            "model_args": action.args, "args": rec["args"], "decision": rec.get("decision"),
            "status": rec.get("status"), "error_type": rec.get("error_type"), "reason": rec.get("reason", ""),
            "attempts": rec.get("attempts"), "retries": rec.get("retries", []),
            "latency_ms": rec.get("latency_ms"),
            "context_items": len(ctx), "observation": obs,
        })

        # --- update the explicit state with the result
        if result is not None:
            if tool == "parse_resume":
                state = state.model_copy(update={"resume_json": result})
                if result["injection_warning"]:
                    trace.append({"phase": "security_alert",
                                  "detail": "Prompt-injection line in the resume was dropped and treated as data."})
            elif tool == "parse_vacancy":
                state = state.model_copy(update={"vacancy_json": result})
            elif tool == "compute_gap":
                state = state.model_copy(update={"gap_json": result})
            elif tool == "search_question_bank":
                skill = args["skill"]
                state = state.model_copy(update={"searched_skills": state.searched_skills + [skill]})
                state = _verify_and_collect(state, skill, result["results"], trace)
            elif tool == "save_prep_plan":
                state = state.model_copy(update={"saved_plan_path": result["file_path"]})
            continue

        # --- stop condition 3: failures the agent must not "work around"
        if tool in ("parse_resume", "parse_vacancy") and rec.get("error_type") in ("invalid_input", "permanent"):
            stop_reason = "clarification_needed"
            break
        if rec.get("error_type") == "transient":
            stop_reason = "tool_failed"
            break
        # permission_denied / invalid model args: the model sees the error and decides again
    return state, stop_reason, model_text


def _final_answer(state: AgentState, stop_reason: str) -> str:
    if stop_reason == "clarification_needed":
        return ("I could not extract enough information. Which technologies have you worked with, "
                "and what are the requirements of the job?")
    if stop_reason == "tool_failed":
        return "The question bank is currently unavailable. Please try again later. Nothing was saved."
    gap = state.gap_json or {}
    lines = []
    if (state.resume_json or {}).get("injection_warning"):
        lines.append("Warning: the resume contained instructions to the agent; they were treated as data and ignored.")
    lines += [
        f"Target role: {(state.vacancy_json or {}).get('role') or 'unknown'}",
        f"Matched skills: {', '.join(gap.get('matched', [])) or 'none'}",
        f"Missing skills (gaps): {', '.join(gap.get('missing', [])) or 'none'}",
        f"Weak skills: {', '.join(gap.get('weak', [])) or 'none'}",
        f"Prepared {len(state.candidate_questions)} grounded questions.",
        f"Saved approved plan to: {state.saved_plan_path}" if state.saved_plan_path
        else "Plan was not saved (user approval required).",
    ]
    if stop_reason == "step_budget_exhausted":
        lines.append(f"Stopped: step budget of {state.max_steps} tool calls was used up.")
    return "\n".join(lines)


def _write_jsonl(path: Path, entries: list[dict[str, Any]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def generate_submission_traces(traces_dir: str = "docs/traces") -> dict[str, Any]:
    """Writes the two required trace files with the MockLLM (deterministic, no API key):
    a normal run, and a run where search_question_bank fails once (simulated 503) and is retried."""
    out = Path(traces_dir)
    out.mkdir(parents=True, exist_ok=True)
    threads, notes = ThreadStore(), NoteStore()

    normal = run_interview_agent("daria", "thread-normal-01", SAMPLE_RESUME, SAMPLE_VACANCY,
                                 threads, notes, llm=MockLLM(), approve_save=True)
    _write_jsonl(out / "trace_normal.jsonl", normal["trace"])

    original_fn = TOOLS["search_question_bank"]
    calls = {"count": 0}

    def flaky_search(**kwargs):  # MOCK failure: the first call raises a simulated 503
        calls["count"] += 1
        if calls["count"] == 1:
            raise TransientError("MOCK: simulated question bank timeout (503)")
        return original_fn(**kwargs)

    TOOLS["search_question_bank"] = flaky_search
    try:
        transient = run_interview_agent("alina", "thread-transient-01", SAMPLE_RESUME, SAMPLE_VACANCY,
                                        threads, notes, llm=MockLLM(), approve_save=True)
    finally:
        TOOLS["search_question_bank"] = original_fn
    _write_jsonl(out / "trace_transient.jsonl", transient["trace"])
    return {"normal": normal, "transient": transient}


def main() -> None:
    parser = argparse.ArgumentParser(description="InterviewPrep Agent - milestone 1")
    parser.add_argument("--live", action="store_true", help="run one conversation with Gemini (needs GEMINI_API_KEY)")
    args = parser.parse_args()
    if args.live:
        llm = make_llm()
        if llm.name != "gemini":
            raise SystemExit("--live needs GEMINI_API_KEY (environment variable or Colab Secrets)")
        out = run_interview_agent("demo-user", "live-01", SAMPLE_RESUME, SAMPLE_VACANCY,
                                  ThreadStore(), NoteStore(), llm=llm, approve_save=False)
        for entry in out["trace"]:
            print(json.dumps(entry, ensure_ascii=False))
        print("\n" + out["answer"])
        return
    generate_submission_traces("docs/traces")
    print("Generated docs/traces/trace_normal.jsonl and docs/traces/trace_transient.jsonl (MockLLM)")


if __name__ == "__main__":
    main()
