# InterviewPrep Agent – Milestone 1 Submission

**Authors:** Daria Zotova & Alina Lukyanov  
**Course:** AI Agents (סוכני בינה מלאכותית), Department of Information Systems, Max Stern Yezreel Valley College  
**Semester:** Fall 2026 (סמסטר א' תשפ"ז)  
**Target Domain:** Junior DevOps / Python Developer  

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/AlinaLu22/interview-prep-agent/blob/main/interview_prep_milestone1.ipynb)


---

## Overview

**InterviewPrep Agent** helps junior DevOps / Python candidates prepare for technical and behavioral job interviews. It compares a resume with a job posting, computes skill gaps, retrieves relevant questions from a curated question bank, verifies them (no discriminatory topics, grounded in the resume/vacancy) and saves the question set only after the user approves it.

This repository implements **Milestone 1**: tool contracts, validation and permissions, timeout and retries, explicit state, memory isolation, full-vs-reduced context comparison, and an agent loop with traces.

The agent loop is driven by a model that chooses the next tool:

| Mode | Model | When |
| :--- | :--- | :--- |
| default | `MockLLM` – **mock model** (explicitly marked, deterministic, no API key, no cost) | tests, trace files, notebook without a key |
| live | `GeminiLLM` – Google Gemini via function calling; `parse_resume`/`parse_vacancy` extract with Gemini JSON output | when `GEMINI_API_KEY` is set (environment variable or Colab Secrets) |

Step budget: 14 (tool calls including retries, plus the model's final decision). The regular scenario of the proposal takes 10 steps.

---

## Repository Structure

```text
.
├── README.md
├── requirements.txt                  # exact versions
├── pytest.ini                        # lets `pytest` find the src package
├── interview_prep_milestone1.ipynb   # Google Colab notebook (clone -> install -> tests -> demo)
├── src/
│   ├── __init__.py
│   ├── tools.py      # 5 tools, contracts (JSON schemas), extraction check, filter, run_tool (timeout, retries)
│   ├── state.py      # AgentState, ThreadStore, NoteStore, build_context, context comparison
│   ├── llm.py        # MockLLM (mock model), GeminiLLM + GeminiExtractor (real model), make_llm()
│   └── agent.py      # agent loop, approval gate, stop conditions, trace generation
├── tests/
│   ├── __init__.py
│   ├── test_tools.py   # valid / invalid / transient for every tool + timeout, permissions, extraction, filter
│   ├── test_state.py   # state, step budget, thread & memory isolation, context comparison
│   ├── test_agent.py   # loop, stop conditions, approval gate, two conversations do not mix
│   └── test_llm.py     # Gemini agent + extractor with a fake client (no network)
├── data/
│   ├── question_bank.json            # synthetic interview question bank
│   └── filter_test_questions.json    # 20 indirect forbidden + 10 legitimate questions (filter test)
└── docs/
    ├── milestone1.md                 # documentation (max 2 pages)
    ├── ai_usage.md                   # AI usage appendix
    └── traces/
        ├── trace_normal.jsonl        # normal run
        └── trace_transient.jsonl     # run with a transient failure that is retried
```

---

## Installation & Running

### 1. Setup (Python 3.11+; Google Colab uses 3.12)
```bash
git clone https://github.com/AlinaLu22/interview-prep-agent.git
cd interview-prep-agent
pip install -r requirements.txt
```

### 2. Run all tests (one command, mock model, no API key, no external service)
```bash
pytest -v
```

### 3. Generate the trace files (mock model)
```bash
python -m src.agent
```
Writes `docs/traces/trace_normal.jsonl` and `docs/traces/trace_transient.jsonl`.

### 4. Run one conversation with the real model (optional)
```bash
export GEMINI_API_KEY="..."          # never commit the key
export GEMINI_MODEL="gemini-2.5-flash" # optional, this is the default
python -m src.agent --live
```

### 5. Google Colab
Open the notebook with the badge above and choose **Runtime → Run all**. The first cell clones the repository and installs `requirements.txt`. Without a key everything runs with the mock model. To use Gemini, add a secret named `GEMINI_API_KEY` in **Colab Secrets** and enable **Notebook access**.

---

## Security & Privacy Notice
* Only synthetic resumes, synthetic job postings and curated interview questions are included.
* The API key is read only from the environment or Colab Secrets; it is not stored in the repository.
* `save_prep_plan` writes only inside `prep_plans/` (ignored by git) and only after user approval.
