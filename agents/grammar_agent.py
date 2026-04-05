"""
Grammar & Language Agent
------------------------
Evaluates the paper's professional tone, syntax quality,
clarity of writing, and grammatical correctness.
"""

import json
from config import get_llm

GRAMMAR_PROMPT = """You are an expert academic language reviewer and editor for top-tier journals.

Your task: Evaluate the grammatical quality, professional tone, and writing clarity of this research paper.

## Paper Text (Introduction + Methodology + Conclusion):
{text}

## Your Analysis Must Cover:
1. **Grammar & Syntax**: Sentence structure, subject-verb agreement, tense consistency.
2. **Professional Tone**: Is the writing formal and appropriate for academic publication?
3. **Clarity**: Are ideas expressed clearly? Any ambiguous or convoluted sentences?
4. **Technical Writing Quality**: Proper use of technical terms, definitions, notation consistency.
5. **Common Issues**: Passive voice overuse, run-on sentences, jargon without explanation.

## Output Format (respond in valid JSON only):
{{
    "rating": "<High | Medium | Low>",
    "overall_assessment": "<1-2 sentence summary of writing quality>",
    "issues": [
        {{
            "type": "<grammar | tone | clarity | technical>",
            "description": "<specific issue found>",
            "severity": "<minor | moderate | major>"
        }}
    ],
    "suggestions": ["<improvement suggestion 1>", "<suggestion 2>"]
}}

Rate as:
- **High**: Publication-ready, minimal issues
- **Medium**: Acceptable but needs revision in some areas
- **Low**: Significant language issues that hinder comprehension

Respond with ONLY the JSON object, no other text.
"""


def analyze_grammar(sections: dict) -> dict:
    """
    Analyze grammar, tone, and writing quality.

    Args:
        sections: Decomposed paper sections dict.

    Returns:
        dict with rating (High/Medium/Low), overall_assessment, issues, suggestions.
    """
    llm = get_llm(temperature=0.1)

    # Combine key sections (skip references — no grammar value there)
    text_parts = []
    for key in ["introduction", "methodology", "results", "conclusion"]:
        section = sections.get(key, "")
        if section and "not found" not in section.lower():
            text_parts.append(f"## {key.title()}\n{section}")

    combined_text = "\n\n".join(text_parts)

    # If combined text is too long, truncate to fit within limits
    if len(combined_text) > 40000:  # ~10k tokens
        combined_text = combined_text[:40000] + "\n\n[... truncated ...]"

    prompt = GRAMMAR_PROMPT.format(text=combined_text)

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
            "rating": "Medium",
            "overall_assessment": content[:500],
            "issues": [{"type": "parsing", "description": "Could not parse structured response.", "severity": "minor"}],
            "suggestions": [],
        }

    # Ensure required fields
    result.setdefault("rating", "Medium")
    result.setdefault("overall_assessment", "No assessment provided.")
    result.setdefault("issues", [])
    result.setdefault("suggestions", [])

    return result
