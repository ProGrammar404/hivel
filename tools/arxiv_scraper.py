"""
arXiv Paper Scraper
-------------------
Extracts paper metadata and full text from an arXiv URL.

Strategy:
1. Parse the arXiv ID from the URL.
2. Use the `arxiv` library for metadata (title, authors, abstract, published date).
3. Fetch full-text HTML from ar5iv.org (clean HTML mirror of arXiv).
4. Fallback: Download PDF and extract text with PyPDF2.
"""

import re
import io
import arxiv
import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader
from typing import Optional


def extract_arxiv_id(url: str) -> str:
    """
    Extract the arXiv paper ID from various URL formats.
    
    Supports:
        - https://arxiv.org/abs/1706.03762
        - https://arxiv.org/pdf/1706.03762
        - https://ar5iv.org/abs/1706.03762
        - 1706.03762 (raw ID)
    """
    # Match patterns like 1706.03762 or 1706.03762v1
    pattern = r'(\d{4}\.\d{4,5}(?:v\d+)?)'
    match = re.search(pattern, url)
    if match:
        return match.group(1)
    raise ValueError(f"Could not extract arXiv ID from: {url}")


def fetch_metadata(arxiv_id: str) -> dict:
    """Fetch paper metadata using the arxiv Python library."""
    client = arxiv.Client()
    search = arxiv.Search(id_list=[arxiv_id])
    results = list(client.results(search))

    if not results:
        raise ValueError(f"No paper found for arXiv ID: {arxiv_id}")

    paper = results[0]
    return {
        "title": paper.title,
        "authors": [author.name for author in paper.authors],
        "abstract": paper.summary,
        "published": str(paper.published.date()),
        "categories": paper.categories,
        "pdf_url": paper.pdf_url,
        "arxiv_id": arxiv_id,
    }


def fetch_full_text_html(arxiv_id: str) -> Optional[str]:
    """
    Fetch full paper text from ar5iv.org (HTML mirror of arXiv).
    Returns cleaned text or None if unavailable.
    """
    url = f"https://ar5iv.labs.arxiv.org/html/{arxiv_id}"
    
    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    # Remove script, style, nav elements
    for tag in soup(["script", "style", "nav", "header", "footer"]):
        tag.decompose()

    # Extract main article content
    article = soup.find("article") or soup.find("div", class_="ltx_page_content")
    if article:
        text = article.get_text(separator="\n", strip=True)
    else:
        text = soup.get_text(separator="\n", strip=True)

    # Clean up excessive whitespace
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    return "\n".join(lines)


def fetch_full_text_pdf(pdf_url: str) -> Optional[str]:
    """
    Fallback: Download PDF and extract text using PyPDF2.
    Returns extracted text or None on failure.
    """
    try:
        response = requests.get(pdf_url, timeout=60)
        response.raise_for_status()
    except requests.RequestException:
        return None

    try:
        reader = PdfReader(io.BytesIO(response.content))
        text_parts = []
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
        return "\n".join(text_parts) if text_parts else None
    except Exception:
        return None


def scrape_paper(url: str) -> dict:
    """
    Main entry point: scrape an arXiv paper from its URL.
    
    Returns a dict with:
        - title, authors, abstract, published, categories
        - full_text: the complete paper text
        - source: "ar5iv" or "pdf" (indicates where text came from)
    """
    # Step 1: Extract ID
    arxiv_id = extract_arxiv_id(url)

    # Step 2: Fetch metadata
    metadata = fetch_metadata(arxiv_id)

    # Step 3: Fetch full text (try HTML first, then PDF)
    full_text = fetch_full_text_html(arxiv_id)
    source = "ar5iv"

    if not full_text:
        full_text = fetch_full_text_pdf(metadata["pdf_url"])
        source = "pdf"

    if not full_text:
        raise RuntimeError(f"Failed to extract text for paper {arxiv_id}")

    return {
        **metadata,
        "full_text": full_text,
        "source": source,
    }


if __name__ == "__main__":
    # Quick test
    test_url = "https://arxiv.org/abs/1706.03762"
    result = scrape_paper(test_url)
    print(f"Title: {result['title']}")
    print(f"Authors: {', '.join(result['authors'][:3])}...")
    print(f"Source: {result['source']}")
    print(f"Text length: {len(result['full_text'])} characters")
    print(f"\nFirst 500 chars:\n{result['full_text'][:500]}")
