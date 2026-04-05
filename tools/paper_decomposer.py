"""
Paper Decomposer
-----------------
Splits raw paper text into logical sections for agent processing.

Sections:
    - Abstract
    - Introduction
    - Methodology (includes Methods, Approach, Model, etc.)
    - Results (includes Experiments, Evaluation, etc.)
    - Conclusion (includes Discussion, Future Work, etc.)
    - References

Also enforces the 16k token limit per chunk.
"""

import re
import tiktoken
from typing import Optional


# Max tokens allowed per LLM call
MAX_TOKENS = 16000

# Reserve some tokens for the prompt template wrapping each section
TOKEN_BUFFER = 2000
MAX_SECTION_TOKENS = MAX_TOKENS - TOKEN_BUFFER

# Common section heading patterns in academic papers
SECTION_PATTERNS = {
    "abstract": [
        r"(?i)^#+\s*abstract",
        r"(?i)^abstract\s*$",
        r"(?i)^abstract\b",
    ],
    "introduction": [
        r"(?i)^#+\s*\d*\.?\s*introduction",
        r"(?i)^\d+\.?\s*introduction",
        r"(?i)^introduction\s*$",
    ],
    "methodology": [
        r"(?i)^#+\s*\d*\.?\s*(?:method|approach|model|architecture|framework|proposed)",
        r"(?i)^\d+\.?\s*(?:method|approach|model|architecture|framework|proposed)",
        r"(?i)^(?:method|approach|model architecture|proposed method|our approach)",
    ],
    "results": [
        r"(?i)^#+\s*\d*\.?\s*(?:result|experiment|evaluation|empirical|performance|finding)",
        r"(?i)^\d+\.?\s*(?:result|experiment|evaluation|empirical|performance|finding)",
        r"(?i)^(?:result|experiment|evaluation|empirical results|findings)",
    ],
    "conclusion": [
        r"(?i)^#+\s*\d*\.?\s*(?:conclusion|discussion|summary|future work|limitation)",
        r"(?i)^\d+\.?\s*(?:conclusion|discussion|summary|future work|limitation)",
        r"(?i)^(?:conclusion|discussion|summary and future|limitations)",
    ],
    "references": [
        r"(?i)^#+\s*references",
        r"(?i)^references\s*$",
        r"(?i)^\[1\]",
    ],
}


def count_tokens(text: str) -> int:
    """Count tokens using tiktoken (cl100k_base encoding, used by most models)."""
    try:
        enc = tiktoken.get_encoding("cl100k_base")
        return len(enc.encode(text))
    except Exception:
        # Rough fallback: ~4 chars per token
        return len(text) // 4


def truncate_to_token_limit(text: str, max_tokens: int = MAX_SECTION_TOKENS) -> str:
    """Truncate text to stay within token limit."""
    tokens = count_tokens(text)
    if tokens <= max_tokens:
        return text

    # Approximate character count for the limit
    ratio = max_tokens / tokens
    char_limit = int(len(text) * ratio * 0.95)  # 5% safety margin
    return text[:char_limit] + "\n\n[... truncated to fit token limit ...]"


def _find_section_start(lines: list[str], patterns: list[str]) -> Optional[int]:
    """Find the line index where a section starts."""
    for i, line in enumerate(lines):
        for pattern in patterns:
            if re.match(pattern, line.strip()):
                return i
    return None


def decompose_paper(full_text: str, abstract: str = "") -> dict:
    """
    Decompose a paper's full text into structured sections.
    
    Args:
        full_text: The complete paper text (from scraper).
        abstract: Pre-extracted abstract from metadata (used as fallback).
    
    Returns:
        dict with keys: abstract, introduction, methodology, results,
                        conclusion, references, full_text
        Each value is a string, truncated to fit within token limits.
    """
    lines = full_text.split("\n")

    # Find section boundaries
    section_starts = {}
    for section_name, patterns in SECTION_PATTERNS.items():
        idx = _find_section_start(lines, patterns)
        if idx is not None:
            section_starts[section_name] = idx

    # Sort sections by their position in the document
    sorted_sections = sorted(section_starts.items(), key=lambda x: x[1])

    # Extract text for each section
    sections = {}
    for i, (name, start) in enumerate(sorted_sections):
        # End is the start of the next section, or end of document
        if i + 1 < len(sorted_sections):
            end = sorted_sections[i + 1][1]
        else:
            end = len(lines)
        
        section_text = "\n".join(lines[start:end]).strip()
        sections[name] = section_text

    # Build the final output with fallbacks
    result = {
        "abstract": sections.get("abstract", abstract or "Abstract not found."),
        "introduction": sections.get("introduction", "Introduction section not found."),
        "methodology": sections.get("methodology", "Methodology section not found."),
        "results": sections.get("results", "Results section not found."),
        "conclusion": sections.get("conclusion", "Conclusion section not found."),
        "references": sections.get("references", "References section not found."),
    }

    # If we couldn't find specific sections, use the full text split into chunks
    found_sections = [k for k, v in result.items() if "not found" not in v.lower()]
    
    if len(found_sections) < 3:
        # Fallback: split the full text into roughly equal parts
        result = _fallback_decompose(full_text, abstract)

    # Enforce token limits on each section
    for key in result:
        result[key] = truncate_to_token_limit(result[key])

    # Also include the full text (truncated) for agents that need broad context
    result["full_text"] = truncate_to_token_limit(full_text)

    # Add token counts for transparency
    result["_token_counts"] = {
        key: count_tokens(val) for key, val in result.items() if key != "_token_counts"
    }

    return result


def _fallback_decompose(full_text: str, abstract: str = "") -> dict:
    """
    Fallback: when section headings can't be detected,
    split the paper into proportional chunks.
    
    Typical paper structure by proportion:
        Abstract: ~5%, Introduction: ~15%, Methodology: ~30%,
        Results: ~30%, Conclusion: ~10%, References: ~10%
    """
    total_len = len(full_text)
    
    # Use metadata abstract if available
    abs_text = abstract if abstract else full_text[:int(total_len * 0.05)]

    # Split remaining text proportionally
    intro_end = int(total_len * 0.15)
    method_end = int(total_len * 0.45)
    results_end = int(total_len * 0.75)
    conclusion_end = int(total_len * 0.90)

    return {
        "abstract": abs_text,
        "introduction": full_text[:intro_end],
        "methodology": full_text[intro_end:method_end],
        "results": full_text[method_end:results_end],
        "conclusion": full_text[results_end:conclusion_end],
        "references": full_text[conclusion_end:],
    }


if __name__ == "__main__":
    # Quick test with sample text
    sample = """
Abstract
This paper presents a new approach to neural machine translation.

1 Introduction
Neural machine translation has seen great progress in recent years.

2 Methodology
We propose a transformer-based architecture that uses self-attention.

3 Results
Our model achieves state-of-the-art results on WMT 2014 English-German.

4 Conclusion
We have shown that attention mechanisms are sufficient for translation.

References
[1] Vaswani et al. Attention is all you need. 2017.
"""
    result = decompose_paper(sample, abstract="This paper presents a new approach.")
    for section, text in result.items():
        if section == "_token_counts":
            print(f"\nToken counts: {text}")
        else:
            print(f"\n--- {section.upper()} ---")
            print(text[:100] + "..." if len(text) > 100 else text)
