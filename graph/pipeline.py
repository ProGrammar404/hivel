"""
LangGraph Pipeline
------------------
Defines the multi-agent workflow as a LangGraph StateGraph.

Flow:
    [scrape] → [decompose] → [consistency, grammar, novelty, factcheck, authenticity] → [generate_report]
                                          ↑ run in parallel ↑
"""

from langgraph.graph import StateGraph, START, END
from graph.state import PaperState
from tools.arxiv_scraper import scrape_paper
from tools.paper_decomposer import decompose_paper


# ─────────────────────────────────────────────
# Node Functions
# ─────────────────────────────────────────────

def scrape_node(state: PaperState) -> dict:
    """Scrape the arXiv paper and populate metadata + full text."""
    try:
        result = scrape_paper(state["arxiv_url"])
        return {
            "title": result["title"],
            "authors": result["authors"],
            "abstract": result["abstract"],
            "published": result["published"],
            "categories": result["categories"],
            "pdf_url": result["pdf_url"],
            "arxiv_id": result["arxiv_id"],
            "full_text": result["full_text"],
            "scrape_source": result["source"],
            "errors": [],
        }
    except Exception as e:
        return {"errors": [f"Scraping failed: {str(e)}"]}


def decompose_node(state: PaperState) -> dict:
    """Decompose the paper into sections."""
    try:
        sections = decompose_paper(state["full_text"], state.get("abstract", ""))
        return {"sections": sections}
    except Exception as e:
        return {"errors": state.get("errors", []) + [f"Decomposition failed: {str(e)}"]}


def consistency_node(state: PaperState) -> dict:
    """Agent: Check if methodology supports the claimed results."""
    from agents.consistency_agent import analyze_consistency
    try:
        result = analyze_consistency(state["sections"])
        return {"consistency_result": result}
    except Exception as e:
        return {
            "consistency_result": {"score": 0, "reasoning": f"Agent error: {str(e)}"},
            "errors": state.get("errors", []) + [f"Consistency agent failed: {str(e)}"],
        }


def grammar_node(state: PaperState) -> dict:
    """Agent: Evaluate grammar, tone, and syntax."""
    from agents.grammar_agent import analyze_grammar
    try:
        result = analyze_grammar(state["sections"])
        return {"grammar_result": result}
    except Exception as e:
        return {
            "grammar_result": {"rating": "Unknown", "issues": [f"Agent error: {str(e)}"]},
            "errors": state.get("errors", []) + [f"Grammar agent failed: {str(e)}"],
        }


def novelty_node(state: PaperState) -> dict:
    """Agent: Assess the novelty of the paper."""
    from agents.novelty_agent import analyze_novelty
    try:
        result = analyze_novelty(state["sections"], state.get("title", ""), state.get("abstract", ""))
        return {"novelty_result": result}
    except Exception as e:
        return {
            "novelty_result": {"novelty_index": "Unknown", "similar_papers": [], "assessment": f"Agent error: {str(e)}"},
            "errors": state.get("errors", []) + [f"Novelty agent failed: {str(e)}"],
        }


def factcheck_node(state: PaperState) -> dict:
    """Agent: Verify cited claims, constants, and formulas."""
    from agents.factcheck_agent import analyze_factcheck
    try:
        result = analyze_factcheck(state["sections"])
        return {"factcheck_result": result}
    except Exception as e:
        return {
            "factcheck_result": {"verified_claims": [], "unverified_claims": [], "log": f"Agent error: {str(e)}"},
            "errors": state.get("errors", []) + [f"Factcheck agent failed: {str(e)}"],
        }


def authenticity_node(state: PaperState) -> dict:
    """Agent: Calculate fabrication probability."""
    from agents.authenticity_agent import analyze_authenticity
    try:
        result = analyze_authenticity(state["sections"])
        return {"authenticity_result": result}
    except Exception as e:
        return {
            "authenticity_result": {"fabrication_probability": 0, "red_flags": [], "assessment": f"Agent error: {str(e)}"},
            "errors": state.get("errors", []) + [f"Authenticity agent failed: {str(e)}"],
        }


def report_node(state: PaperState) -> dict:
    """Generate the final Markdown Judgement Report."""
    from report.report_generator import generate_report
    try:
        report = generate_report(state)
        return {"final_report": report}
    except Exception as e:
        return {
            "final_report": f"# Error\n\nReport generation failed: {str(e)}",
            "errors": state.get("errors", []) + [f"Report generation failed: {str(e)}"],
        }


# ─────────────────────────────────────────────
# Fan-out / Fan-in for Parallel Agent Execution
# ─────────────────────────────────────────────

def route_to_agents(state: PaperState) -> list[str]:
    """After decomposition, fan out to all 5 agents in parallel."""
    return ["consistency", "grammar", "novelty", "factcheck", "authenticity"]


# ─────────────────────────────────────────────
# Build the Graph
# ─────────────────────────────────────────────

def build_pipeline() -> StateGraph:
    """Construct and compile the LangGraph pipeline."""

    graph = StateGraph(PaperState)

    # Add nodes
    graph.add_node("scrape", scrape_node)
    graph.add_node("decompose", decompose_node)
    graph.add_node("consistency", consistency_node)
    graph.add_node("grammar", grammar_node)
    graph.add_node("novelty", novelty_node)
    graph.add_node("factcheck", factcheck_node)
    graph.add_node("authenticity", authenticity_node)
    graph.add_node("report", report_node)

    # Define edges: sequential start
    graph.add_edge(START, "scrape")
    graph.add_edge("scrape", "decompose")

    # Fan-out: decompose → all 5 agents in parallel
    graph.add_conditional_edges(
        "decompose",
        route_to_agents,
        ["consistency", "grammar", "novelty", "factcheck", "authenticity"],
    )

    # Fan-in: all 5 agents → report
    graph.add_edge("consistency", "report")
    graph.add_edge("grammar", "report")
    graph.add_edge("novelty", "report")
    graph.add_edge("factcheck", "report")
    graph.add_edge("authenticity", "report")

    # Report → END
    graph.add_edge("report", END)

    return graph.compile()


# Pre-built compiled pipeline for easy import
pipeline = build_pipeline()
