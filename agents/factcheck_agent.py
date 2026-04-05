"""
Fact-Check Agent
----------------
Verifies cited claims, constants, formulas, and historical data
mentioned in the paper. Identifies verifiable vs. unverifiable claims.
"""

import json
from config import get_llm

FACTCHECK_PROMPT = """You are a meticulous fact-checker for academic research papers. Your job is to identify and verify factual claims made in the paper.

## Paper Results Section:
{results}

## Paper Introduction (for cited facts/claims):
{introduction}

## Paper Conclusion:
{conclusion}

## Your Analysis Must Cover:
1. **Identify Factual Claims**: Find all verifiable claims (numbers, statistics, cited facts, mathematical constants, historical references, benchmark comparisons).
2. **Verify Each Claim**: For each claim, assess whether it appears accurate based on your knowledge.
3. **Flag Unverifiable Claims**: Claims that cannot be verified without access to the original data.
4. **Check Citations**: Are cited results from other papers accurately represented?
5. **Mathematical/Statistical Claims**: Are formulas, equations, or statistical statements correct?

## Output Format (respond in valid JSON only):
{{
    "verified_claims": [
        {{
            "claim": "<the factual claim>",
            "verdict": "Verified",
            "confidence": "<High | Medium | Low>",
            "explanation": "<why this is considered verified>"
        }}
    ],
    "unverified_claims": [
        {{
            "claim": "<the factual claim>",
            "verdict": "Unverified",
            "reason": "<why this could not be verified>",
            "risk_level": "<Low | Medium | High>"
        }}
    ],
    "flagged_claims": [
        {{
            "claim": "<the potentially incorrect claim>",
            "verdict": "Potentially Incorrect",
            "explanation": "<why this seems wrong or misleading>"
        }}
    ],
    "log": "<summary paragraph of the fact-checking process and overall findings>"
}}

Respond with ONLY the JSON object, no other text.
"""


def analyze_factcheck(sections: dict) -> dict:
    """
    Fact-check the paper's claims, constants, and cited data.

    Args:
        sections: Decomposed paper sections dict.

    Returns:
        dict with verified_claims, unverified_claims, flagged_claims, log.
    """
    llm = get_llm(temperature=0.1)

    prompt = FACTCHECK_PROMPT.format(
        results=sections.get("results", "Not available."),
        introduction=sections.get("introduction", "Not available."),
        conclusion=sections.get("conclusion", "Not available."),
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
            "verified_claims": [],
            "unverified_claims": [],
            "flagged_claims": [],
            "log": content[:500],
        }

    # Ensure required fields
    result.setdefault("verified_claims", [])
    result.setdefault("unverified_claims", [])
    result.setdefault("flagged_claims", [])
    result.setdefault("log", "No log provided.")

    return result
