"""
Consistency Agent
-----------------
Checks whether the paper's methodology actually supports the claimed results.
Looks for logical gaps, unsupported conclusions, and mismatches between
what was proposed and what was evaluated.
"""

import json
from config import get_llm

CONSISTENCY_PROMPT = """You are a senior academic peer reviewer specializing in methodological consistency analysis.

Your task: Analyze whether the paper's METHODOLOGY actually supports and leads to the claimed RESULTS.

## Paper Methodology Section:
{methodology}

## Paper Results Section:
{results}

## Paper Abstract (for context):
{abstract}

## Your Analysis Must Cover:
1. **Alignment**: Do the experiments described in Results directly test what the Methodology proposes?
2. **Logical Flow**: Are there logical gaps between the method and the conclusions drawn from results?
3. **Missing Links**: Are there claims in the results that have no corresponding methodological basis?
4. **Statistical Rigor**: Do the results seem properly derived from the described methodology?
5. **Overclaiming**: Does the paper claim more than what the methodology can support?

## Output Format (respond in valid JSON only):
{{
    "score": <integer 0-100, where 100 = perfectly consistent>,
    "reasoning": "<detailed paragraph explaining the consistency analysis>",
    "strengths": ["<strength 1>", "<strength 2>"],
    "weaknesses": ["<weakness 1>", "<weakness 2>"],
    "alignment_issues": ["<issue 1 if any>"]
}}

Respond with ONLY the JSON object, no other text.
"""


def analyze_consistency(sections: dict) -> dict:
    """
    Analyze consistency between methodology and results.

    Args:
        sections: Decomposed paper sections dict.

    Returns:
        dict with score (0-100), reasoning, strengths, weaknesses, alignment_issues.
    """
    llm = get_llm(temperature=0.1)

    prompt = CONSISTENCY_PROMPT.format(
        methodology=sections.get("methodology", "Not available."),
        results=sections.get("results", "Not available."),
        abstract=sections.get("abstract", "Not available."),
    )

    response = llm.invoke(prompt)
    content = response.content.strip()

    # Parse JSON from response (handle markdown code blocks)
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
        content = content.strip()

    try:
        result = json.loads(content)
    except json.JSONDecodeError:
        result = {
            "score": 50,
            "reasoning": content,
            "strengths": [],
            "weaknesses": ["Could not parse structured response."],
            "alignment_issues": [],
        }

    # Ensure required fields exist
    result.setdefault("score", 50)
    result.setdefault("reasoning", "No reasoning provided.")
    result.setdefault("strengths", [])
    result.setdefault("weaknesses", [])
    result.setdefault("alignment_issues", [])

    return result
