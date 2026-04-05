"""
Hivel — Agentic Research Paper Evaluator
=========================================
Streamlit UI that accepts an arXiv URL, runs the multi-agent
LangGraph pipeline, and displays a rendered Markdown Judgement Report.
"""

import re
import time
import streamlit as st

# ─── Page Config ───
st.set_page_config(
    page_title="Hivel — Paper Evaluator",
    page_icon="📋",
    layout="wide",
)

# ─── Node display metadata (order matters) ───
NODE_META = {
    "scrape":       ("📥", "Scraping paper from arXiv…"),
    "decompose":    ("🔪", "Decomposing paper into sections…"),
    "consistency":  ("🔗", "Agent 1/5 — Analyzing methodology ↔ results consistency…"),
    "grammar":      ("✍️",  "Agent 2/5 — Evaluating grammar & language quality…"),
    "novelty":      ("🔬", "Agent 3/5 — Assessing novelty & searching related work…"),
    "factcheck":    ("🔍", "Agent 4/5 — Fact-checking claims & references…"),
    "authenticity": ("🛡️",  "Agent 5/5 — Detecting potential fabrication…"),
    "report":       ("📋", "Generating Judgement Report…"),
}

TOTAL_STEPS = len(NODE_META)


def validate_arxiv_url(url: str) -> bool:
    """Check if the URL looks like a valid arXiv link."""
    patterns = [
        r"arxiv\.org/abs/\d{4}\.\d{4,5}",
        r"arxiv\.org/pdf/\d{4}\.\d{4,5}",
        r"ar5iv\.labs\.arxiv\.org/html/\d{4}\.\d{4,5}",
        r"^\d{4}\.\d{4,5}$",  # bare ID
    ]
    return any(re.search(p, url.strip()) for p in patterns)


def normalize_url(url: str) -> str:
    """Normalize input to a standard arXiv abs URL."""
    url = url.strip()
    # Bare ID → full URL
    if re.match(r"^\d{4}\.\d{4,5}(v\d+)?$", url):
        return f"https://arxiv.org/abs/{url}"
    return url


# ─── Header ───
st.markdown(
    """
    <h1 style='text-align: center;'>📋 Hivel</h1>
    <p style='text-align: center; color: grey; margin-top: -10px;'>
        Agentic Research Paper Evaluator — powered by LangGraph + Gemini
    </p>
    <hr>
    """,
    unsafe_allow_html=True,
)

# ─── Sidebar ───
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("---")

    arxiv_input = st.text_input(
        "🔗 arXiv URL or Paper ID",
        placeholder="e.g. https://arxiv.org/abs/1706.03762",
        help="Paste any arXiv abs/pdf URL or just the paper ID like 1706.03762",
    )

    st.markdown("---")
    st.markdown(
        """
        **How it works:**
        1. Scrapes the paper from arXiv
        2. Decomposes it into sections
        3. Runs **5 specialized AI agents**:
           - 🔗 Consistency
           - ✍️ Grammar
           - 🔬 Novelty
           - 🔍 Fact-Check
           - 🛡️ Authenticity
        4. Generates a **Judgement Report**
        """
    )
    st.markdown("---")
    st.caption("Built with LangGraph · Gemini · Streamlit")

# ─── Main Area ───
col_left, col_right = st.columns([1, 2])

with col_left:
    analyze_btn = st.button("🚀 Analyze Paper", use_container_width=True, type="primary")

# ─── Run Pipeline ───
if analyze_btn:
    if not arxiv_input or not arxiv_input.strip():
        st.error("Please enter an arXiv URL or paper ID.")
        st.stop()

    if not validate_arxiv_url(arxiv_input):
        st.error("That doesn't look like a valid arXiv URL or ID. Try something like `https://arxiv.org/abs/1706.03762` or just `1706.03762`.")
        st.stop()

    url = normalize_url(arxiv_input)

    # Import pipeline lazily (avoids loading LangGraph on every page refresh)
    from graph.pipeline import pipeline

    # Progress UI
    progress_bar = st.progress(0, text="Initializing pipeline…")
    status_container = st.container()

    step_count = 0
    start_time = time.time()
    final_state = {}

    try:
        # stream() yields {node_name: output_dict} for each completed node
        for chunk in pipeline.stream({"arxiv_url": url}):
            node_name = list(chunk.keys())[0]
            node_output = chunk[node_name]

            # Merge into final_state
            if isinstance(node_output, dict):
                final_state.update(node_output)

            step_count += 1
            progress = step_count / TOTAL_STEPS

            icon, msg = NODE_META.get(node_name, ("⚡", f"Running {node_name}…"))

            # Show completed step
            with status_container:
                elapsed = time.time() - start_time
                st.success(f"{icon} **{node_name.capitalize()}** completed  ({elapsed:.0f}s)")

            # Update progress bar with next step info
            if step_count < TOTAL_STEPS:
                # Peek at what's next
                next_nodes = list(NODE_META.keys())
                if step_count < len(next_nodes):
                    next_icon, next_msg = NODE_META[next_nodes[step_count]]
                    progress_bar.progress(progress, text=f"{next_icon} {next_msg}")
            else:
                progress_bar.progress(1.0, text="✅ All steps complete!")

    except Exception as e:
        st.error(f"Pipeline failed: {str(e)}")
        st.stop()

    elapsed_total = time.time() - start_time

    # ─── Display Results ───
    st.markdown("---")

    report = final_state.get("final_report", "")
    errors = final_state.get("errors", [])

    if errors:
        with st.expander(f"⚠️ {len(errors)} warning(s) during analysis", expanded=False):
            for err in errors:
                st.warning(err)

    if report:
        # Stats row
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            score = final_state.get("consistency_result", {}).get("score", "—")
            st.metric("🔗 Consistency", f"{score}/100")
        with col2:
            rating = final_state.get("grammar_result", {}).get("rating", "—")
            st.metric("✍️ Grammar", rating)
        with col3:
            novelty = final_state.get("novelty_result", {}).get("novelty_index", "—")
            st.metric("🔬 Novelty", novelty)
        with col4:
            fab = final_state.get("authenticity_result", {}).get("fabrication_probability", "—")
            risk = final_state.get("authenticity_result", {}).get("risk_level", "")
            st.metric("🛡️ Fabrication Risk", f"{fab}%", delta=risk, delta_color="inverse")

        st.markdown("---")

        # Rendered report
        st.markdown(report, unsafe_allow_html=False)

        st.markdown("---")

        # Download buttons
        col_dl1, col_dl2, _ = st.columns([1, 1, 2])
        with col_dl1:
            st.download_button(
                label="📥 Download Report (.md)",
                data=report,
                file_name=f"judgement_report_{final_state.get('arxiv_id', 'paper')}.md",
                mime="text/markdown",
                use_container_width=True,
            )
        with col_dl2:
            # Also offer a plain text version
            st.download_button(
                label="📄 Download Report (.txt)",
                data=report,
                file_name=f"judgement_report_{final_state.get('arxiv_id', 'paper')}.txt",
                mime="text/plain",
                use_container_width=True,
            )

        st.caption(f"⏱️ Total analysis time: {elapsed_total:.0f}s")
    else:
        st.error("No report was generated. Check the errors above.")

# ─── Empty State ───
elif not arxiv_input:
    st.markdown(
        """
        <div style='text-align: center; padding: 60px 20px; color: grey;'>
            <h2>👈 Enter an arXiv URL to get started</h2>
            <p>Paste any arXiv paper link or ID in the sidebar, then click <b>Analyze Paper</b>.</p>
            <br>
            <p style='font-size: 0.9em;'>
                Example: <code>https://arxiv.org/abs/1706.03762</code> (Attention Is All You Need)
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
