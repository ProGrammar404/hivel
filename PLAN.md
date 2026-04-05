# Hivel — Plan

## Overview

Multi-agent system that takes an arXiv paper URL, scrapes the content, runs it through 5 specialized evaluation agents, and produces a structured Judgement Report.

---

## Architecture

```
[arXiv URL]
    │
    ▼
[Scraper] ── fetch paper text (ar5iv HTML / PDF fallback)
    │
    ▼
[Decomposer] ── split into sections (Abstract, Methodology, Results, Conclusion, References)
    │
    ▼
┌───────────────────────────────────────────────┐
│           LangGraph Pipeline                  │
│                                               │
│  ┌──────────────┐  ┌──────────┐  ┌─────────┐ │
│  │ Consistency   │  │ Grammar  │  │ Novelty │ │
│  │ Agent         │  │ Agent    │  │ Agent   │ │
│  └──────┬───────┘  └────┬─────┘  └────┬────┘ │
│         │               │              │      │
│  ┌──────┴───────┐  ┌────┴─────────┐   │      │
│  │ Fact-Check   │  │ Authenticity │   │      │
│  │ Agent        │  │ Agent        │   │      │
│  └──────┬───────┘  └────┬─────────┘   │      │
│         │               │              │      │
└─────────┼───────────────┼──────────────┼──────┘
          ▼               ▼              ▼
    ┌─────────────────────────────────────────┐
    │         Report Generator                │
    │   (Markdown Judgement Report)            │
    └─────────────────────────────────────────┘
          │
          ▼
    [Streamlit UI]
```

---

## Agents

**Consistency Agent**
- Input: Methodology + Results
- Output: `{ score: 0-100, reasoning, strengths, weaknesses, alignment_issues }`
- Checks if methodology actually supports the claimed results

**Grammar Agent**
- Input: Introduction + Methodology + Results + Conclusion
- Output: `{ rating: High/Medium/Low, overall_assessment, issues, suggestions }`
- Evaluates professional tone, syntax, and clarity

**Novelty Agent**
- Input: Title + Abstract + Introduction
- Searches arXiv for similar existing papers
- Output: `{ novelty_index, core_contribution, similar_work_comparison, assessment }`

**Fact-Check Agent**
- Input: Results + Introduction + Conclusion
- Output: `{ verified_claims, unverified_claims, flagged_claims, log }`
- Verifies cited claims, constants, formulas, benchmark numbers

**Authenticity Agent**
- Input: All sections
- Output: `{ fabrication_probability: 0-100, risk_level, red_flags, positive_indicators }`
- Detects signs of data fabrication, p-hacking, statistical anomalies

---

## Report Output

Structured Markdown with:
- Executive Summary (Pass / Conditional Pass / Fail)
- Consistency Score (0–100)
- Grammar Rating (High / Medium / Low)
- Novelty Index
- Fact-Check Log (verified vs unverified claims)
- Fabrication Risk Score (%)

---

## Tech Stack

| Component | Choice |
|-----------|--------|
| Agentic Framework | LangGraph |
| LLM | Gemini 2.0 Flash (free tier) |
| Paper Scraping | `arxiv` lib + ar5iv.org + PyPDF2 |
| Token Counting | tiktoken |
| UI | Streamlit |
| Report Format | Markdown |

---

## Constraints

- No single LLM call exceeds 16k tokens — enforced by section-aware chunking
- Sequential agent execution to stay within Gemini free-tier rate limits (15 RPM)
- Each agent only receives the sections it actually needs
