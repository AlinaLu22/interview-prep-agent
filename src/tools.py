"""Tools of InterviewPrep Agent: explicit contracts, implementations and the run_tool wrapper.

Every tool has a contract (name, description, input schema, output schema, side effect,
approval requirement). run_tool is the only way the agent executes a tool:

    unknown tool? -> input validation -> permission check -> execute with timeout
    -> retry only on TransientError (exponential backoff) -> output validation -> record

The tool names follow the tools table (section 6) of the approved project proposal.
"""
import json
import re
import time
import unicodedata
from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from pathlib import Path
from typing import Any, Callable

from jsonschema import ValidationError as SchemaError
from jsonschema import validate


class TransientError(Exception):
    """Temporary failure (timeout, 429, 503, file temporarily unavailable): worth retrying."""


class PermanentError(Exception):
    """Permanent failure (no data to extract, corrupted file): retrying will not help."""


PROJECT_ROOT = Path(__file__).resolve().parent.parent
BANK_PATH = PROJECT_ROOT / "data" / "question_bank.json"
# The ONLY folder save_prep_plan is allowed to write into (checked before execution).
PLANS_ROOT = PROJECT_ROOT / "prep_plans"

# ==============================================================================
# QUESTION BANK (loaded lazily, so an unavailable file is a retryable tool error,
# not a crash at import time)
# ==============================================================================

_BANK_CACHE: list[dict[str, Any]] | None = None


def load_question_bank() -> list[dict[str, Any]]:
    global _BANK_CACHE
    if _BANK_CACHE is None:
        try:
            raw = BANK_PATH.read_text(encoding="utf-8")
        except OSError as e:
            raise TransientError(f"question bank unavailable: {e}") from e
        try:
            _BANK_CACHE = json.loads(raw)
        except json.JSONDecodeError as e:
            raise PermanentError(f"question bank file is corrupted: {e}") from e
    return _BANK_CACHE


# ==============================================================================
# SKILL EXTRACTION (deterministic keyword dictionary, see "known limitations")
# ==============================================================================

WORD_RE = re.compile(r"\w+")


def tokenize(text: str) -> set[str]:
    """Lower-case words of 2+ letters, as a set."""
    return {w.lower() for w in WORD_RE.findall(text) if len(w) > 1}


SKILLS_DICT: dict[str, list[str]] = {
    "python": ["python"],
    "sql": ["sql", "database"],
    "docker": ["docker", "container", "containers", "dockerfile"],
    "ci/cd": ["ci", "cd", "cicd", "pipeline", "jenkins", "github actions", "gitlab"],
    "linux": ["linux", "bash", "systemd", "ubuntu"],
    "git": ["git", "github", "gitlab"],
    "networking": ["network", "networking", "ccna", "tcp", "dns", "http"],
    "aws": ["aws", "cloud", "ec2", "s3", "iam"],
    "security": ["security", "hardening", "siem", "ssh"],
    "llm ai agent": ["llm", "ai", "agent", "agents", "rag"],
    "teamwork": ["teamwork", "team", "collaboration"],
    "leadership": ["led", "lead", "leadership", "manage"],
    "learning": ["learning", "learn"],
}

YEARS_RE = re.compile(r"(\d+(?:\.\d+)?)\s*\+?\s*(?:years?|yrs?)\b", re.I)
BASIC_RE = re.compile(r"\b(basic|beginner|familiar|exposure|introductory)\b", re.I)


def _level_from(text: str, years: float) -> str:
    if "junior" in text.lower() or years < 2:
        return "junior"
    return "mid" if years < 5 else "senior"


SKILL_NAMES = list(SKILLS_DICT)  # canonical skill names: the extractor must answer with these


def _extract_skills_with_source(text: str) -> list[dict[str, Any]]:
    """Keyword extraction used by the MOCK extractor. One entry per skill: the first line that
    mentions it is the source line. Years come from that line ("1.5 years"); "basic"/"familiar" caps at 0.5."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    found = []
    for skill, synonyms in SKILLS_DICT.items():
        syn_set = set(synonyms)
        for ln in lines:
            if tokenize(ln) & syn_set or any(s in ln.lower() for s in synonyms if " " in s):
                m = YEARS_RE.search(ln)
                years = float(m.group(1)) if m else 1.0
                if BASIC_RE.search(ln):
                    years = min(years, 0.5)
                found.append({
                    "skill": skill,
                    "level": _level_from(text, years),
                    "years": years,
                    "source_line": ln,
                })
                break
    return found


# ==============================================================================
# EXTRACTION BY THE MODEL (proposal section 4, step 2: the model extracts, the code validates)
# parse_resume / parse_vacancy ask the configured extractor for JSON. By default that is the
# MOCK extractor below; with GEMINI_API_KEY the agent switches to GeminiExtractor (src/llm.py).
# ==============================================================================

NICE_TO_HAVE_WORDS = ("advantage", "plus", "nice to have", "bonus", "רצוי")


class KeywordExtractor:
    """MOCK MODEL for extraction (מודל מדומה, מסומן במפורש): deterministic keyword dictionary,
    no API key, no cost. Returns the same JSON shape that the real model is asked to return."""

    name = "mock-extractor"

    def extract_resume(self, text: str) -> dict[str, Any]:
        return {"skills": _extract_skills_with_source(text)}

    def extract_vacancy(self, text: str) -> dict[str, Any]:
        role = next((ln.strip().rstrip(".:") for ln in text.splitlines() if ln.strip()), "")
        reqs = [{"skill": s["skill"], "source_line": s["source_line"],
                 "required": not any(w in s["source_line"].lower() for w in NICE_TO_HAVE_WORDS)}
                for s in _extract_skills_with_source(text)]
        return {"role": role, "requirements": reqs}


_EXTRACTOR: Any = KeywordExtractor()


@contextmanager
def use_extractor(extractor: Any):
    """Runs the block with the given extractor (e.g. the Gemini one) and restores the previous one."""
    global _EXTRACTOR
    previous, _EXTRACTOR = _EXTRACTOR, extractor
    try:
        yield
    finally:
        _EXTRACTOR = previous


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def _grounded(items: Any, text: str) -> list[dict[str, Any]]:
    """Code-side check of the model's JSON: a known skill, and a source line that really appears
    in the input text. Anything else is dropped, so the agent cannot invent skills."""
    if not isinstance(items, list):
        raise PermanentError("extractor returned invalid JSON (expected a list)")
    haystack = _squash(text)
    kept, seen = [], set()
    for it in items:
        if not isinstance(it, dict):
            continue
        skill = str(it.get("skill", "")).strip().lower()
        src = str(it.get("source_line", "")).strip()
        if skill in SKILLS_DICT and src and _squash(src) in haystack and skill not in seen:
            seen.add(skill)
            kept.append({**it, "skill": skill, "source_line": src[:140]})
    return kept


# ==============================================================================
# TWO-LAYER QUESTION FILTER (proposal section 4, step 6)
# Layer 1: forbidden (discriminatory) topics, phrased as whole words/phrases so that
#          technical terms ("health check", "git push origin", "race condition",
#          "child process", Hebrew "למדת", "רגיל") are NOT blocked.
# Layer 2: grounding - the question must relate to a skill from the resume/vacancy.
# ==============================================================================

# Layer 1: regex patterns and phrase lists per forbidden category (proposal step 6), in English and
# Hebrew, matched on normalized text. Whole words/phrases only, so technical terms such as
# "health check", "git push origin", "race condition", "child process", "למדת", "רגיל" are not blocked.
FORBIDDEN_EN: dict[str, re.Pattern] = {
    "age": re.compile(r"\b(how old are you|your age|date of birth|birth\s*year|year of birth|when were you born)\b"),
    "family_pregnancy": re.compile(
        r"\b(married|marital status|spouse|pregnan\w*|maternity|family plans|family planning"
        r"|start a family|grow your family|family (?:is )?(?:planning|plans) to grow|baby on the way"
        r"|have (?:any )?(?:kids|children)|your (?:kids|children)"
        r"|plan(?:ning)? to (?:get married|have (?:kids|children)))\b"),
    "religion_holidays": re.compile(r"\b(religio\w*|church|synagogue|mosque|pray\w*|shabbat|sabbath"
                                    r"|holidays do you (?:celebrate|observe))\b"),
    "health": re.compile(r"\b(health (?:condition|issue|problem|history)s?|medical (?:condition|history|issue)s?"
                         r"|disabilit\w*|chronic (?:illness|disease|condition)s?|illness\w*|mental health"
                         r"|sick leave)\b"),
    "origin_residence": re.compile(
        r"\b(country of origin|ethnic\w*|nationality|citizenship|your race|race(?!\s+conditions?)"
        r"|where (?:are|were) you (?:originally )?from|where did you grow up"
        r"|where do you live|your (?:home )?address|which (?:city|town|neighbou?rhood) do you live"
        r"|how far (?:away )?do you live|live (?:far|close|near)(?: from| to)?(?: the)? office)\b"),
}


def _he(*alternatives: str) -> re.Pattern:
    # A whole Hebrew word/phrase with at most ONE attached prefix letter (ו/ה/ב/ל/מ/ש/כ).
    return re.compile(r"(?<!\w)[והבלמשכ]?(?:" + "|".join(alternatives) + r")(?!\w)")


FORBIDDEN_HE: dict[str, re.Pattern] = {
    "age": _he("בן כמה", "בת כמה", "גיל", "גילך", "גילכם", "שנת לידה", "תאריך לידה", "נולדת"),
    "family_pregnancy": _he("מצב משפחתי", "ילדים", "ילדיך", "ילדייך", "הריון", "נשוי", "נשואה", "רווק", "רווקה",
                            "גרוש", "גרושה", r"התחת[נן]\w*", r"תתחת[נן]\w*", "חתונה", "נישואין",
                            "תכנון משפחה", "להרחיב את המשפחה"),
    "religion_holidays": _he("דת", "דתי", "דתייה", "דתית", "חגים", r"שומר(?:ת)? שבת", "בית כנסת", "כנסייה",
                             "מסגד", r"מתפלל\w*"),
    "health": _he(r"בריאות\w*", "מחלה", "מחלות", "נכות", "תרופות"),
    "origin_residence": _he(r"ארץ (?:ה)?מוצא", "מוצאך", "מוצאכם", "מוצא עדתי", r"מאיפה (?:אתה|את) במקור",
                            "איפה גדלת", "לאום", r"איפה (?:אתה|את) גר(?:ה)?", r"באיזו עיר (?:אתה|את) גר(?:ה)?",
                            "מקום מגורים", "כתובת מגורים", "כתובתך"),
}

FORBIDDEN_CATEGORIES = list(FORBIDDEN_EN)
FORBIDDEN_PATTERNS = list(FORBIDDEN_EN.values()) + list(FORBIDDEN_HE.values())


def normalize_text(text: str) -> str:
    """Lower case, no niqqud, no punctuation, single spaces (proposal step 6, layer 1)."""
    nfkd = unicodedata.normalize("NFKD", text)
    no_niqqud = "".join(c for c in nfkd if not unicodedata.combining(c))
    cleaned = re.sub(r"[^\w\s֐-׿]", " ", no_niqqud.lower())
    return re.sub(r"\s+", " ", cleaned).strip()


def forbidden_category(question: str) -> str | None:
    norm = normalize_text(question)
    for category in FORBIDDEN_CATEGORIES:
        if FORBIDDEN_EN[category].search(norm) or FORBIDDEN_HE[category].search(norm):
            return category
    return None


def check_question_safety(
    question: str,
    allowed_skills: set[str],
    source_line: str = "",
    context_text: str = "",
) -> tuple[bool, str]:
    """Layer 1: forbidden category. Layer 2 (grounding): the question must relate to a skill or
    requirement from the resume/vacancy and have a source line - so a family or age question in a
    new phrasing, with no source skill, is blocked too. context_text = topic of the bank entry."""
    if forbidden_category(question):
        return False, "forbidden_topic_layer1"
    if not source_line.strip():
        return False, "missing_source_grounding_layer2"
    q_words = tokenize(normalize_text(question)) | tokenize(normalize_text(context_text))
    skill_tokens: set[str] = set()
    for s in allowed_skills:
        skill_tokens |= tokenize(s)
        skill_tokens |= set(SKILLS_DICT.get(s, []))
    if not (q_words & skill_tokens):
        return False, "unanchored_question_layer2"
    return True, "ok"


# ==============================================================================
# 5 CONTRACTS FROM THE APPROVED PROPOSAL (section 6, tools table)
# ==============================================================================

PARSE_RESUME_CONTRACT = {
    "name": "parse_resume",
    "description": "Extract structured skills (skill, level, years, source line) from synthetic resume text "
                   "with the model; the code validates the JSON and keeps only skills whose source line is "
                   "really in the text. Lines that look like prompt injection are dropped and reported.",
    "type": "model (extraction) + schema validation in code",  # as in the proposal tools table
    "timeout_s": 120,  # calls the model (with its own retries on rate limits)
    "side_effect": "none",
    "requires_approval": False,
    "input_schema": {
        "type": "object",
        "properties": {"resume_text": {"type": "string", "minLength": 15, "maxLength": 8000}},
        "required": ["resume_text"],
        "additionalProperties": False,
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "skills": {
                "type": "array",
                "minItems": 1,
                "items": {
                    "type": "object",
                    "properties": {
                        "skill": {"type": "string"},
                        "level": {"type": "string", "enum": ["junior", "mid", "senior"]},
                        "years": {"type": "number", "minimum": 0},
                        "source_line": {"type": "string"},
                    },
                    "required": ["skill", "level", "years", "source_line"],
                },
            },
            "injection_warning": {"type": "boolean"},
        },
        "required": ["skills", "injection_warning"],
    },
}

PARSE_VACANCY_CONTRACT = {
    "name": "parse_vacancy",
    "description": "Extract the role title, mandatory (must_have) and optional (nice_to_have) "
                   "requirements and their source lines from a job description with the model; "
                   "the code validates the JSON and the source lines.",
    "type": "model (extraction) + schema validation in code",  # as in the proposal tools table
    "timeout_s": 120,  # calls the model (with its own retries on rate limits)
    "side_effect": "none",
    "requires_approval": False,
    "input_schema": {
        "type": "object",
        "properties": {"vacancy_text": {"type": "string", "minLength": 15, "maxLength": 8000}},
        "required": ["vacancy_text"],
        "additionalProperties": False,
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "role": {"type": "string"},
            "must_have": {"type": "array", "items": {"type": "string"}},
            "nice_to_have": {"type": "array", "items": {"type": "string"}},
            "source_lines": {"type": "object"},
        },
        "required": ["role", "must_have", "nice_to_have", "source_lines"],
    },
}

COMPUTE_GAP_CONTRACT = {
    "name": "compute_gap",
    "description": "Compare parsed resume skills with parsed vacancy requirements: matched, "
                   "missing (not in resume) and weak (in resume with less than 1 year).",
    "type": "computation",  # as in the proposal tools table
    "side_effect": "none",
    "requires_approval": False,
    "input_schema": {
        "type": "object",
        "properties": {
            "resume_json": {"type": "object", "required": ["skills"]},
            "vacancy_json": {"type": "object", "required": ["must_have", "nice_to_have"]},
        },
        "required": ["resume_json", "vacancy_json"],
        "additionalProperties": False,
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "matched": {"type": "array", "items": {"type": "string"}},
            "missing": {"type": "array", "items": {"type": "string"}},
            "weak": {"type": "array", "items": {"type": "string"}},
        },
        "required": ["matched", "missing", "weak"],
    },
}

SEARCH_BANK_CONTRACT = {
    "name": "search_question_bank",
    "description": "Read-only search of the local interview question bank by skill, level, "
                   "category and count. Returns an empty list when nothing matches.",
    "type": "read",  # as in the proposal tools table
    "side_effect": "read",
    "requires_approval": False,
    "input_schema": {
        "type": "object",
        "properties": {
            "skill": {"type": "string", "minLength": 2, "maxLength": 100},
            "level": {"type": "string", "enum": ["junior", "mid", "senior"]},
            "top_k": {"type": "integer", "minimum": 1, "maximum": 5},
            "category": {"type": "string", "enum": ["technical", "behavioral"]},
        },
        "required": ["skill"],
        "additionalProperties": False,
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "results": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "doc_id": {"type": "string"},
                        "topic": {"type": "string"},
                        "category": {"type": "string"},
                        "level": {"type": "string"},
                        "question": {"type": "string"},
                        "look_for": {"type": "string"},
                        "score": {"type": "number"},
                    },
                    "required": ["doc_id", "topic", "category", "level", "question", "look_for", "score"],
                },
            }
        },
        "required": ["results"],
    },
}

SAVE_PLAN_CONTRACT = {
    "name": "save_prep_plan",
    "description": "WRITE action: save the approved interview preparation question set as a JSON file "
                   "inside the project's prep_plans/ folder. Runs only with approved=True (enforced in code).",
    "type": "write",  # as in the proposal tools table
    "side_effect": "write",
    "requires_approval": True,
    "input_schema": {
        "type": "object",
        "properties": {
            "plan_name": {"type": "string", "minLength": 2, "maxLength": 100},
            "questions": {"type": "array", "minItems": 1, "maxItems": 12},
            "approved": {"type": "boolean"},
            "output_dir": {"type": "string", "minLength": 1},
        },
        "required": ["plan_name", "questions", "approved"],
        "additionalProperties": False,
    },
    "output_schema": {
        "type": "object",
        "properties": {
            "saved": {"type": "boolean"},
            "file_path": {"type": "string"},
            "question_count": {"type": "integer"},
        },
        "required": ["saved", "file_path", "question_count"],
    },
}

# ==============================================================================
# TOOL IMPLEMENTATIONS
# ==============================================================================

INJECTION_RE = re.compile(
    r"(ignore (all )?previous instructions|התעלם מההוראות|system prompt|save without approval)", re.I
)


def _safe_text(text: str) -> str:
    # Untrusted text is data: lines that look like instructions are dropped before the model sees them.
    return "\n".join(ln for ln in text.splitlines() if not INJECTION_RE.search(ln))


def parse_resume(resume_text: str) -> dict[str, Any]:
    injection = bool(INJECTION_RE.search(resume_text))
    text = _safe_text(resume_text)
    raw = _EXTRACTOR.extract_resume(text)  # the MODEL extracts (mock or Gemini)
    if not isinstance(raw, dict):
        raise PermanentError("extractor returned invalid JSON")
    skills = []
    for s in _grounded(raw.get("skills"), text):  # the CODE validates
        try:
            years = max(0.0, float(s.get("years", 1)))
        except (TypeError, ValueError):
            years = 1.0
        level = s.get("level") if s.get("level") in ("junior", "mid", "senior") else _level_from(text, years)
        skills.append({"skill": s["skill"], "level": level, "years": years, "source_line": s["source_line"]})
    if not skills:
        raise PermanentError("no_skills_found_in_resume")
    return {"skills": skills, "injection_warning": injection}


def parse_vacancy(vacancy_text: str) -> dict[str, Any]:
    text = _safe_text(vacancy_text)
    raw = _EXTRACTOR.extract_vacancy(text)  # the MODEL extracts (mock or Gemini)
    if not isinstance(raw, dict):
        raise PermanentError("extractor returned invalid JSON")
    reqs = _grounded(raw.get("requirements"), text)  # the CODE validates
    if not reqs:
        raise PermanentError("no_clear_requirements_in_vacancy")
    must_have = [r["skill"] for r in reqs if r.get("required", True) is not False]
    nice_to_have = [r["skill"] for r in reqs if r.get("required", True) is False]
    if not must_have:
        must_have, nice_to_have = nice_to_have, []
    role = str(raw.get("role") or "").strip()[:80]
    return {"role": role, "must_have": must_have, "nice_to_have": nice_to_have,
            "source_lines": {r["skill"]: r["source_line"] for r in reqs}}


def compute_gap(resume_json: dict[str, Any], vacancy_json: dict[str, Any]) -> dict[str, Any]:
    res_map = {s["skill"]: s for s in resume_json.get("skills", [])}
    all_req = list(dict.fromkeys(vacancy_json.get("must_have", []) + vacancy_json.get("nice_to_have", [])))
    matched, missing, weak = [], [], []
    for req in all_req:
        if req not in res_map:
            missing.append(req)
        elif res_map[req].get("years", 1) < 1:
            weak.append(req)
        else:
            matched.append(req)
    return {"matched": matched, "missing": missing, "weak": weak}


def search_question_bank(
    skill: str, level: str = "junior", top_k: int = 3, category: str | None = None
) -> dict[str, Any]:
    bank = load_question_bank()
    q_words = tokenize(skill) | set(SKILLS_DICT.get(skill.lower(), []))
    results = []
    for doc in bank:
        if category and doc["category"] != category:
            continue
        if doc.get("level", level) != level:
            continue
        doc_words = tokenize(" ".join([doc["topic"], doc["keywords"], doc["question"]]))
        overlap = len(q_words & doc_words)
        if overlap:
            results.append({
                "doc_id": doc["id"],
                "topic": doc["topic"],
                "category": doc["category"],
                "level": doc.get("level", level),
                "question": doc["question"],
                "look_for": doc["look_for"],
                "score": round(overlap / max(1, len(q_words)), 2),
            })
    results.sort(key=lambda r: r["score"], reverse=True)
    return {"results": results[: int(top_k)]}


def save_prep_plan(
    plan_name: str, questions: list[dict[str, Any]], approved: bool, output_dir: str | None = None
) -> dict[str, Any]:
    if approved is not True:  # second line of defence; run_tool already blocks this
        raise PermissionError("Approval flag is False: saving prep plan is blocked by code.")
    out_path = Path(output_dir) if output_dir else PLANS_ROOT
    out_path.mkdir(parents=True, exist_ok=True)
    safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", plan_name)
    file_path = out_path / f"{safe_name}.json"
    payload = {"plan_name": plan_name, "approved": True, "question_count": len(questions), "questions": questions}
    file_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    try:  # project-relative path: no local machine paths in logs and traces
        shown = file_path.resolve().relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        shown = str(file_path)
    return {"saved": True, "file_path": shown, "question_count": len(questions)}


# ==============================================================================
# EXECUTION WRAPPER
# ==============================================================================

TOOLS: dict[str, Callable[..., dict[str, Any]]] = {
    "parse_resume": parse_resume,
    "parse_vacancy": parse_vacancy,
    "compute_gap": compute_gap,
    "search_question_bank": search_question_bank,
    "save_prep_plan": save_prep_plan,
}

CONTRACTS: dict[str, dict[str, Any]] = {
    "parse_resume": PARSE_RESUME_CONTRACT,
    "parse_vacancy": PARSE_VACANCY_CONTRACT,
    "compute_gap": COMPUTE_GAP_CONTRACT,
    "search_question_bank": SEARCH_BANK_CONTRACT,
    "save_prep_plan": SAVE_PLAN_CONTRACT,
}

DEFAULT_TIMEOUT_S = 5.0
_EXECUTOR = ThreadPoolExecutor(max_workers=4, thread_name_prefix="tool")

KEY_ORDER = ["step", "tool", "args", "decision", "reason", "error_type",
             "attempts", "retries", "status", "latency_ms", "timeout_s"]

# error_type values: unknown_tool | invalid_input | permission_denied | transient | permanent


def check_permissions(name: str, args: dict[str, Any]) -> str | None:
    """Returns a denial reason, or None when the call is allowed. Runs BEFORE execution."""
    contract = CONTRACTS[name]
    if contract.get("requires_approval") and args.get("approved") is not True:
        return "approved flag is False"
    if name == "save_prep_plan" and "output_dir" in args:
        root = PLANS_ROOT.resolve()
        target = Path(args["output_dir"]).resolve()
        if target != root and root not in target.parents:
            return f"output_dir is outside the allowed folder ({root})"
    return None


def _short(value: Any) -> Any:
    """Keeps logs readable and free of raw resume text."""
    if isinstance(value, str) and len(value) > 60:
        return f"<text: {len(value)} chars>"
    if isinstance(value, list) and len(value) > 3:
        return f"<list: {len(value)} items>"
    if isinstance(value, dict) and len(json.dumps(value, ensure_ascii=False)) > 120:
        return f"<object: keys {sorted(value)}>"
    return value


def _call_with_timeout(fn: Callable[..., Any], args: dict[str, Any], timeout_s: float) -> Any:
    future = _EXECUTOR.submit(fn, **args)
    try:
        return future.result(timeout=timeout_s)
    except FutureTimeout:
        future.cancel()  # a Python thread cannot be killed; the late result is ignored
        raise TransientError(f"timeout after {timeout_s}s") from None


def run_tool(
    name: str,
    args: dict[str, Any],
    max_attempts: int = 2,
    base_delay: float = 0.2,
    timeout_s: float | None = None,
    step: int | None = None,
    verbose: bool = False,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    """Returns (result or None, execution record). The record is plain JSON and is stored
    by the caller in the state of the conversation that made the call (no global log)."""
    record: dict[str, Any] = {"step": step, "tool": name, "args": {k: _short(v) for k, v in args.items()}}

    def finish(result: dict[str, Any] | None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
        clean = {k: record[k] for k in KEY_ORDER if k in record}
        if verbose:
            print(json.dumps(clean, ensure_ascii=False))
        return result, clean

    if name not in TOOLS:
        record.update(decision="rejected", reason=f"unknown tool '{name}'", error_type="unknown_tool",
                      attempts=0, status="error", latency_ms=0)
        return finish(None)

    contract = CONTRACTS[name]
    try:
        validate(instance=args, schema=contract["input_schema"])
    except SchemaError as e:
        record.update(decision="rejected", reason=f"invalid args: {e.message}", error_type="invalid_input",
                      attempts=0, status="error", latency_ms=0)
        return finish(None)

    denied = check_permissions(name, args)
    if denied:
        record.update(decision="rejected", reason=f"permission_denied: {denied}", error_type="permission_denied",
                      attempts=0, status="error", latency_ms=0)
        return finish(None)

    timeout = float(timeout_s or contract.get("timeout_s", DEFAULT_TIMEOUT_S))
    record.update(decision="executed", timeout_s=timeout)
    start = time.perf_counter()
    result: dict[str, Any] | None = None
    attempt = 0
    for attempt in range(1, max_attempts + 1):
        try:
            result = _call_with_timeout(TOOLS[name], args, timeout)
        except TransientError as e:
            result = None
            record.setdefault("retries", []).append(f"attempt {attempt}: {e}")
            if attempt < max_attempts:
                time.sleep(base_delay * (2 ** (attempt - 1)))
                continue
            record.update(status="error", error_type="transient",
                          reason=f"transient error after {attempt} attempts: {e}")
            break
        except Exception as e:  # PermanentError, PermissionError, bugs: never retried
            result = None
            record.update(status="error", error_type="permanent",
                          reason=f"permanent failure: {type(e).__name__}: {e}")
            break
        try:
            validate(instance=result, schema=contract["output_schema"])
        except SchemaError as e:
            result = None
            record.update(status="error", error_type="permanent",
                          reason=f"output violates contract: {e.message}")
            break
        record.update(status="ok")
        break

    record["attempts"] = attempt
    record["latency_ms"] = round((time.perf_counter() - start) * 1000)
    return finish(result)
