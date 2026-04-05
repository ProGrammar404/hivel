# Hivel — Agentic Research Paper Evaluator

Multi-agent system that scrapes arXiv papers and generates a peer-review style Judgement Report using LangGraph and Gemini.

---

## What it does

Takes an arXiv URL, scrapes the paper, decomposes it into sections, and runs 5 evaluation agents:

| Agent | Evaluates | Output |
|-------|-----------|--------|
| Consistency | Methodology vs Results alignment | Score 0–100 |
| Grammar | Writing quality, tone, syntax | High / Medium / Low |
| Novelty | Originality, searches arXiv for related work | Novelty Index |
| Fact-Check | Verifies cited claims, constants, formulas | Verified / Unverified / Flagged |
| Authenticity | Fabrication detection, statistical anomalies | Risk 0–100% |

Outputs a Markdown report with an Executive Summary (Pass / Conditional Pass / Fail).

---

## Architecture

```
[arXiv URL]
    │
    ▼
┌──────────┐     ┌──────────────┐
│ Scraper  │────▶│ Decomposer   │
│ (ar5iv/  │     │ (split into  │
│  PDF)    │     │  sections)   │
└──────────┘     └──────┬───────┘
                        │
          ┌─────────────┼─────────────┐
          ▼             ▼             ▼
   ┌────────────┐ ┌──────────┐ ┌───────────┐
   │Consistency │ │ Grammar  │ │  Novelty  │
   │  Agent     │ │  Agent   │ │  Agent    │
   └─────┬──────┘ └────┬─────┘ └─────┬─────┘
         │              │             │
   ┌─────┴──────┐ ┌────┴──────┐      │
   │ Fact-Check │ │Authenticity│      │
   │  Agent     │ │  Agent    │      │
   └─────┬──────┘ └────┬──────┘      │
         │              │             │
         └──────────────┼─────────────┘
                        ▼
              ┌──────────────────┐
              │ Report Generator │
              │  (Markdown)      │
              └────────┬─────────┘
                       ▼
              ┌──────────────────┐
              │  Streamlit UI    │
              └──────────────────┘
```

Agents run sequentially to stay within Gemini free-tier rate limits. Each LLM call is capped at 16k tokens.

---

## Project Structure

```
hivel/
├── agents/
│   ├── consistency_agent.py
│   ├── grammar_agent.py
│   ├── novelty_agent.py
│   ├── factcheck_agent.py
│   └── authenticity_agent.py
├── tools/
│   ├── arxiv_scraper.py
│   └── paper_decomposer.py
├── graph/
│   ├── state.py
│   └── pipeline.py
├── report/
│   └── report_generator.py
├── app.py
├── config.py
├── requirements.txt
├── sample_report.md
├── PLAN.md
└── .env                        # you create this
```

---

## Setup

### Prerequisites

- Python 3.12+
- A Google Gemini API key — get one free at [aistudio.google.com/apikey](https://aistudio.google.com/apikey)

### Install

```bash
git clone https://github.com/ProGrammar404/hivel.git
cd hivel
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Configure API Key

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_google_gemini_api_key_here
```

This file is gitignored and won't be committed. The app won't start without it.

### Run

```bash
streamlit run app.py
```

Opens at `http://localhost:8501`. Paste an arXiv URL, click Analyze.

### CLI Usage

```python
from graph.pipeline import pipeline

result = pipeline.invoke({"arxiv_url": "https://arxiv.org/abs/1706.03762"})
print(result["final_report"])
```

---

## Streamlit UI

- Sidebar: paste arXiv URL or paper ID (e.g. `1706.03762`)
- Live progress bar showing each agent's status
- Metric cards: Consistency, Grammar, Novelty, Fabrication Risk
- Full rendered Markdown report
- Download as `.md` or `.txt`

---

## Sample Output

See [`sample_report.md`](sample_report.md) — generated for "Attention Is All You Need" (1706.03762).

| Metric | Result |
|--------|--------|
| Consistency | 95/100 |
| Grammar | High |
| Novelty | Highly Novel |
| Fabrication Risk | 5% (Very Low) |
| Verdict | PASS |

---

## Config

Editable in `config.py`:

| Setting | Default | Description |
|---------|---------|-------------|
| `MODEL_NAME` | `gemini-2.0-flash` | Gemini model |
| `MAX_TOKENS_PER_CALL` | `16000` | Token limit per LLM call |
| `MIN_DELAY_BETWEEN_CALLS` | `5` | Seconds between API calls |

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Agentic Framework | LangGraph |
| LLM | Google Gemini (free tier) |
| Paper Scraping | arxiv + ar5iv.org + PyPDF2 |
| Token Counting | tiktoken |
| UI | Streamlit |

---

## Design Notes

- **Sequential execution** — Gemini free tier is 15 RPM; running agents in parallel would cause rate limit errors.
- **ar5iv.org first** — provides clean HTML of arXiv papers, much better than raw PDF extraction. PDF is a fallback.
- **Section-aware prompting** — each agent only gets the sections it needs (e.g. Consistency gets Methodology + Results), keeping calls well under 16k tokens.
- **Graceful error handling** — if any agent fails, the rest still run and the report shows what succeeded.

---

## License

MIT
