"""Explicit application state, per-conversation storage, long-term memory and context building."""
import math
import re
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


MAX_STEPS = 14  # proposal section 9: measured 10 steps in the regular scenario + 4 reserve


class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: str


class AgentState(BaseModel):
    """Explicit application state of ONE conversation (one thread), as in proposal section 7.

    step counts the budget of the proposal (section 9): every tool call the model chose, every retry,
    and the model's final decision; code checks and the user's approval are not counted.
    tool_log keeps the execution records of THIS thread only."""
    model_config = ConfigDict(extra="forbid")

    thread_id: str
    user_id: str
    step: int = 0
    max_steps: int = MAX_STEPS
    retry_count: int = 0
    # raw inputs of this run only: cleared when the run ends (never copied to NoteStore)
    resume_text: str = ""
    vacancy_text: str = ""
    # structured results of the tools
    resume_json: dict | None = None
    vacancy_json: dict | None = None
    gap_json: dict | None = None
    searched_skills: list[str] = Field(default_factory=list)
    candidate_questions: list[dict] = Field(default_factory=list)
    # approval comes ONLY from the user at the approval gate, never from the model
    approved: bool = False
    saved_plan_path: str | None = None
    tool_log: list[dict] = Field(default_factory=list)
    messages: list[Message] = Field(default_factory=list)
    summary: str = ""


def apply_event(state: AgentState, role: Literal["user", "assistant", "tool"], content: str) -> AgentState:
    """Pure state transition: returns a NEW AgentState copy (the old one is unchanged).
    A "tool" event is one agent action and is counted against the step budget."""
    if role == "tool" and state.step >= state.max_steps:
        raise RuntimeError("step budget exhausted")
    msgs = state.messages + [Message(role=role, content=content)]
    new_step = state.step + (1 if role == "tool" else 0)
    return state.model_copy(update={"messages": msgs, "step": new_step})


class ThreadStore:
    """thread_id -> AgentState, with user_id ownership enforced in code."""

    def __init__(self) -> None:
        self._threads: dict[str, AgentState] = {}

    def get_or_create(self, thread_id: str, user_id: str, max_steps: int = MAX_STEPS) -> AgentState:
        state = self._threads.get(thread_id)
        if state is None:
            state = AgentState(thread_id=thread_id, user_id=user_id, max_steps=max_steps)
            self._threads[thread_id] = state
        if state.user_id != user_id:
            raise PermissionError("thread belongs to another user")
        return state

    def save(self, state: AgentState) -> None:
        self.get_or_create(state.thread_id, state.user_id)
        self._threads[state.thread_id] = state


def _words(text: str) -> set[str]:
    return {w for w in re.findall(r"\w+", text.lower()) if len(w) > 2}


class NoteStore:
    """Cross-conversation long-term memory, namespaced strictly by user_id. The agent writes here
    only short facts of an APPROVED, saved prep plan (proposal section 7), never raw text."""

    def __init__(self) -> None:
        self._notes: dict[str, list[str]] = {}

    def put(self, user_id: str, text: str) -> None:
        notes = self._notes.setdefault(user_id, [])
        if text not in notes:  # the same fact is stored once
            notes.append(text)

    def search(self, user_id: str, query: str, k: int = 3) -> list[str]:
        q = _words(query)
        scored = [(len(q & _words(n)), n) for n in self._notes.get(user_id, [])]
        scored.sort(key=lambda p: -p[0])
        return [n for hits, n in scored[:k] if hits > 0]

    def all(self, user_id: str) -> list[str]:
        return list(self._notes.get(user_id, []))

    def delete_all(self, user_id: str) -> int:
        return len(self._notes.pop(user_id, []))


class SaveNoteArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    text: str = Field(min_length=3, max_length=300)


class RecallNotesArgs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    query: str = Field(min_length=2, max_length=200)
    k: int = Field(default=3, ge=1, le=5)


SENSITIVE_RE = re.compile(
    r"password|passwd|парол|סיסמ|\b\d{9}\b|\S+@\S+\.\S+|\+?\d[\d\- ]{8,}\d",
    re.IGNORECASE,
)


def run_memory_tool(store: NoteStore, session_user_id: str, name: str, raw_args: dict) -> dict:
    """save_note / recall_notes bound strictly to session_user_id (never model-supplied)."""
    try:
        if name == "save_note":
            args = SaveNoteArgs(**raw_args)
            if SENSITIVE_RE.search(args.text):
                return {"status": "error", "error": {"type": "sensitive_data", "message": "sensitive data blocked"}}
            store.put(session_user_id, args.text)
            return {"status": "ok", "data": {"saved": True}}
        if name == "recall_notes":
            args = RecallNotesArgs(**raw_args)
            return {"status": "ok", "data": {"notes": store.search(session_user_id, args.query, args.k)}}
        return {"status": "error", "error": {"type": "permission_denied", "message": f"{name} not allowed"}}
    except ValidationError as e:
        return {"status": "error", "error": {"type": "invalid_input", "message": str(e)}}


INTERVIEW_SYSTEM = (
    "You are InterviewPrep Agent for Junior DevOps / Python roles. "
    "Use parse_resume, parse_vacancy, compute_gap and search_question_bank to prepare 8-12 grounded "
    "interview questions. Never invent candidate skills, never ask discriminatory questions, "
    "and never save a plan without user approval."
)


def approx_tokens(text: str) -> int:
    return math.ceil(len(text) / 4)


def build_context(
    state: AgentState,
    notes: NoteStore | None = None,
    strategy: str = "notes",
    last_n: int = 4,
    budget: int | None = None,
    query: str = "",
    summarizer: Callable[[list[Message]], str] | None = None,
    system_prompt: str = INTERVIEW_SYSTEM,
) -> list[str]:
    """Builds the exact prompt payload sent to the model: full, last_n, notes (+last_n) or summary."""
    ctx = [system_prompt]
    msgs = state.messages
    if strategy == "full":
        chosen = msgs
    elif strategy == "last_n":
        chosen = msgs[-last_n:]
    elif strategy == "notes":
        found = notes.search(state.user_id, query, 3) if notes else []
        ctx += [f"[note] {n}" for n in found]
        chosen = msgs[-last_n:]
    elif strategy == "summary":
        if summarizer is None:
            raise ValueError("strategy 'summary' requires a summarizer")
        older = msgs[:-last_n]
        if older:
            ctx.append(f"[summary] {summarizer(older)}")
        chosen = msgs[-last_n:]
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    lines = [f"{m.role}: {m.content}" for m in chosen]
    if budget is not None:
        used = sum(approx_tokens(c) for c in ctx)
        kept = []
        for line in reversed(lines):
            t = approx_tokens(line)
            if used + t > budget:
                break
            kept.append(line)
            used += t
        lines = list(reversed(kept))
    return ctx + lines


def context_tokens(ctx: list[str]) -> int:
    return sum(approx_tokens(c) for c in ctx)


# ==============================================================================
# FULL HISTORY vs REDUCED CONTEXT: one question, asked once per strategy
# ==============================================================================

COMPARISON_QUESTION = "Remind me which role am I preparing for, and which skills am I missing?"
COMPARISON_SYSTEM = (
    "You are an interview preparation assistant. Answer the user's last message using ONLY the "
    "conversation and notes below. If they do not contain the answer, say that you do not know."
)
# fact -> spellings accepted in the answer
EXPECTED_FACTS: dict[str, tuple[str, ...]] = {
    "role (junior devops)": ("junior devops",),
    "missing: docker": ("docker",),
    "missing: ci/cd": ("ci/cd", "ci cd", "cicd"),
}


def build_comparison_conversation() -> tuple[AgentState, NoteStore]:
    """A 15-message preparation conversation: the key facts are said only at the start."""
    s = AgentState(thread_id="cmp-1", user_id="daria")
    s = apply_event(s, "user", "Preparing for Junior DevOps. Known: Python, Linux. Missing: Docker, CI/CD.")
    s = apply_event(s, "assistant", "Understood. Focus on Docker and CI/CD.")
    for i in range(6):
        s = apply_event(s, "user", f"Practice question {i} on Linux permissions and bash.")
        s = apply_event(s, "assistant", f"Explanation {i} covering chmod, chown, and systemd.")
    s = apply_event(s, "user", COMPARISON_QUESTION)
    mem = NoteStore()
    run_memory_tool(mem, "daria", "save_note", {"text": "Target role: Junior DevOps"})
    run_memory_tool(mem, "daria", "save_note", {"text": "Missing skills: docker, ci/cd"})
    return s, mem


def _facts_in(text: str) -> list[str]:
    low = text.lower()
    return [fact for fact, spellings in EXPECTED_FACTS.items() if any(sp in low for sp in spellings)]


def compare_context_strategies(llm: Any, last_n: int = 2) -> list[dict[str, Any]]:
    """Sends the same question to the model with each context strategy and records the
    context size, the model's answer, whether the answer is correct and what was lost.
    `llm` must provide answer(context_lines) -> str (MockLLM or GeminiLLM)."""
    state, mem = build_comparison_conversation()
    strategies = [
        ("full", {"strategy": "full"}),
        (f"last_n ({last_n})", {"strategy": "last_n", "last_n": last_n}),
        (f"notes + last_n ({last_n})", {"strategy": "notes", "last_n": last_n, "notes": mem,
                                        "query": "missing skills target role"}),
    ]
    rows = []
    for name, kw in strategies:
        # neutral system prompt: it must not leak the role, otherwise every strategy "knows" it
        ctx = build_context(state, system_prompt=COMPARISON_SYSTEM, **kw)
        answer = llm.answer(ctx)
        in_answer = _facts_in(answer)
        in_context = _facts_in("\n".join(ctx))
        rows.append({
            "strategy": name,
            "model": getattr(llm, "name", type(llm).__name__),
            "messages_in_context": sum(1 for c in ctx if c.split(":", 1)[0] in ("user", "assistant", "tool")),
            "context_items": len(ctx),
            "context_chars": len("\n".join(ctx)),
            "approx_tokens": context_tokens(ctx),
            "answer": answer.strip(),
            "correct": len(in_answer) == len(EXPECTED_FACTS),
            "lost_in_context": [f for f in EXPECTED_FACTS if f not in in_context],
        })
    return rows
