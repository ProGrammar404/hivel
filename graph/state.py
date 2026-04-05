"""
LangGraph State Definition
--------------------------
Central state object that flows through the entire pipeline.
Each node reads from and writes to this state.
"""

import operator
from typing import Annotated, TypedDict, Optional


class PaperState(TypedDict, total=False):
    """
    State that flows through the LangGraph pipeline.

    Fields are grouped by pipeline stage:
        1. Input
        2. Scraping
        3. Decomposition
        4. Agent Results
        5. Final Output
    """

    # ── 1. Input ──
    arxiv_url: str

    # ── 2. Scraping ──
    title: str
    authors: list[str]
    abstract: str
    published: str
    categories: list[str]
    pdf_url: str
    arxiv_id: str
    full_text: str
    scrape_source: str  # "ar5iv" or "pdf"

    # ── 3. Decomposition ──
    sections: dict  # {abstract, introduction, methodology, results, conclusion, references, full_text, _token_counts}

    # ── 4. Agent Results ──
    consistency_result: dict    # {score: 0-100, reasoning: str}
    grammar_result: dict        # {rating: High/Medium/Low, issues: list}
    novelty_result: dict        # {novelty_index: str, similar_papers: list, assessment: str}
    factcheck_result: dict      # {verified_claims: list, unverified_claims: list, log: str}
    authenticity_result: dict   # {fabrication_probability: 0-100, red_flags: list, assessment: str}

    # ── 5. Final Output ──
    final_report: str           # Generated Markdown report

    # ── Error Tracking ──
    errors: Annotated[list[str], operator.add]  # Collect any errors (supports parallel writes)
