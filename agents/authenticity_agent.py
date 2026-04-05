"""
Authenticity Agent
------------------
Calculates a "Fabrication Probability" score by looking for:
- Statistical anomalies
- Logical leaps
- Too-good-to-be-true results
- Inconsistent data patterns
- Signs of data fabrication or p-hacking
"""

import json
from config import get_llm

AUTHENTICITY_PROMPT = """You are a research integrity specialist. Your task is to assess the authenticity and detect potential signs of fabrication in a research paper.

## Paper Abstract:
{abstract}

## Paper Methodology:
{methodology}

## Paper Results:
{results}

## Paper Conclusion:
{conclusion}

## Your Analysis Must Cover:
1. **Statistical Anomalies**: Are results suspiciously perfect or rounded? Do distributions look natural?
2. **Logical Leaps**: Are there conclusions drawn without adequate supporting evidence?
3. **Too-Good-to-Be-True Results**: Do improvements seem unrealistically large compared to baselines?
4. **Data Consistency**: Do numbers add up? Are tables and text consistent?
5. **Methodology Transparency**: Is the methodology described in enough detail to be reproduced?
6. **Red Flags**: Missing error bars, no ablation studies, cherry-picked metrics, etc.

## Output Format (respond in valid JSON only):
{{
    "fabrication_probability": <integer 0-100, where 0 = highly authentic and 100 = likely fabricated>,
    "risk_level": "<Very Low | Low | Moderate | High | Very High>",
    "red_flags": [
        {{
            "flag": "<description of the red flag>",
            "severity": "<minor | moderate | critical>",
            "evidence": "<specific text or data that raised this flag>"
        }}
    ],
    "positive_indicators": ["<indicator of authenticity 1>", "<indicator 2>"],
    "reproducibility_assessment": "<Can this work be reproduced based on the described methodology?>",
    "assessment": "<detailed paragraph summarizing the authenticity analysis>"
}}

IMPORTANT: A low fabrication_probability (0-20) means the paper appears genuine. Only flag high scores (60+) if there are clear red flags. Most legitimate papers should score 5-30.

Respond with ONLY the JSON object, no other text.
"""


def analyze_authenticity(sections: dict) -> dict:
    """
    Assess paper authenticity and calculate fabrication probability.

    Args:
        sections: Decomposed paper sections dict.

    Returns:
        dict with fabrication_probability (0-100), red_flags, assessment, etc.
    """
    llm = get_llm(temperature=0.1)

    prompt = AUTHENTICITY_PROMPT.format(
        abstract=sections.get("abstract", "Not available."),
        methodology=sections.get("methodology", "Not available."),
        results=sections.get("results", "Not available."),
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
            "fabrication_probability": 25,
            "risk_level": "Unknown",
            "red_flags": [],
            "positive_indicators": [],
            "reproducibility_assessment": "Could not assess.",
            "assessment": content[:500],
        }

    # Ensure required fields
    result.setdefault("fabrication_probability", 25)
    result.setdefault("risk_level", "Unknown")
    result.setdefault("red_flags", [])
    result.setdefault("positive_indicators", [])
    result.setdefault("reproducibility_assessment", "Not assessed.")
    result.setdefault("assessment", "No assessment provided.")

    return result
