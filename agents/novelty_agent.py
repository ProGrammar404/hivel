"""
Novelty Agent
-------------
Assesses the novelty and uniqueness of the paper by:
1. Analyzing the claims of contribution.
2. Searching arXiv for similar existing papers.
3. Comparing the paper's approach to found literature.
"""

import json
import arxiv
from config import get_llm

NOVELTY_PROMPT = """You are a senior researcher tasked with evaluating the novelty of a research paper.

## Paper Title:
{title}

## Paper Abstract:
{abstract}

## Paper Introduction (claims of contribution):
{introduction}

## Similar Existing Papers Found on arXiv:
{similar_papers}

## Your Analysis Must Cover:
1. **Core Contribution**: What does this paper claim as its main novel contribution?
2. **Prior Work Overlap**: How much does this overlap with the similar papers listed above?
3. **Incremental vs. Breakthrough**: Is this an incremental improvement or a fundamentally new idea?
4. **Unique Aspects**: What aspects (if any) are genuinely novel?
5. **Novelty Rating**: Overall assessment of uniqueness.

## Output Format (respond in valid JSON only):
{{
    "novelty_index": "<Highly Novel | Moderately Novel | Incremental | Low Novelty>",
    "core_contribution": "<1-2 sentence description of the paper's main claimed contribution>",
    "overlap_with_existing": "<description of overlap with similar papers>",
    "unique_aspects": ["<unique aspect 1>", "<unique aspect 2>"],
    "similar_work_comparison": [
        {{
            "paper": "<title of similar paper>",
            "relationship": "<how it relates to the reviewed paper>"
        }}
    ],
    "assessment": "<detailed paragraph explaining the novelty assessment>"
}}

Respond with ONLY the JSON object, no other text.
"""


def _search_similar_papers(title: str, abstract: str, max_results: int = 5) -> list[dict]:
    """Search arXiv for papers similar to the one being reviewed."""
    query = title

    client = arxiv.Client()
    search = arxiv.Search(
        query=query,
        max_results=max_results + 1,  # +1 to account for the paper itself
        sort_by=arxiv.SortCriterion.Relevance,
    )

    similar = []
    for paper in client.results(search):
        # Skip the paper itself
        if paper.title.strip().lower() == title.strip().lower():
            continue
        similar.append({
            "title": paper.title,
            "authors": ", ".join(a.name for a in paper.authors[:3]),
            "published": str(paper.published.date()),
            "abstract": paper.summary[:300] + "...",
        })
        if len(similar) >= max_results:
            break

    return similar


def analyze_novelty(sections: dict, title: str = "", abstract: str = "") -> dict:
    """
    Analyze the novelty of the paper.

    Args:
        sections: Decomposed paper sections dict.
        title: Paper title (from metadata).
        abstract: Paper abstract (from metadata).

    Returns:
        dict with novelty_index, similar_papers, assessment, etc.
    """
    llm = get_llm(temperature=0.2)

    # Use metadata abstract if available, otherwise from sections
    if not abstract:
        abstract = sections.get("abstract", "")

    # Search for similar papers
    similar_papers = _search_similar_papers(title, abstract)

    # Format similar papers for the prompt
    if similar_papers:
        similar_text = "\n".join(
            f"- **{p['title']}** ({p['published']}) by {p['authors']}\n  {p['abstract']}"
            for p in similar_papers
        )
    else:
        similar_text = "No similar papers found on arXiv."

    prompt = NOVELTY_PROMPT.format(
        title=title or "Unknown",
        abstract=abstract or sections.get("abstract", "Not available."),
        introduction=sections.get("introduction", "Not available."),
        similar_papers=similar_text,
    )

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse JSON from response
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "novelty_index": "Unknown",
            "core_contribution": "",
            "overlap_with_existing": "",
            "unique_aspects": [],
            "similar_work_comparison": [],
            "assessment": content[:500],
        }

    # Attach the raw similar papers list for reference
    result["similar_papers_found"] = similar_papers
    result.setdefault("novelty_index", "Unknown")
    result.setdefault("assessment", "No assessment provided.")
    result.setdefault("unique_aspects", [])

    return result
