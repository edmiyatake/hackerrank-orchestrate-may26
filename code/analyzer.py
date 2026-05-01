"""
analyzer.py — Phase 3: Ticket analyzer (LLM call #1).

Takes a sanitized issue + top retrieved chunks and returns structured analysis:
    - request_type:     product_issue | feature_request | bug | invalid
    - risk_level:       HIGH | MEDIUM | LOW
    - grounded_summary: one-sentence summary grounded in the corpus

The LLM is instructed to ONLY use the provided documentation.
All output is validated via Pydantic before being returned.
Temperature is set to 0 for determinism.

Public API:
    analyze(issue, top_chunks, domain) -> AnalysisResult
"""

from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from typing import Optional
from dotenv import load_dotenv

import anthropic

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

load_dotenv()

ALLOWED_REQUEST_TYPES = {"product_issue", "feature_request", "bug", "invalid"}
ALLOWED_RISK_LEVELS   = {"HIGH", "MEDIUM", "LOW"}
MODEL = "claude-sonnet-4-6"

# Risk keywords that should always trigger HIGH regardless of LLM output
HIGH_RISK_KEYWORDS = [
    # Visa / financial
    "fraud", "stolen", "chargeback", "dispute", "unauthorized",
    "money", "transaction", "stolen card", "lost card",
    # Account access
    "locked out", "can't log in", "cannot login", "account access",
    "suspended", "banned", "hacked", "compromised",
    # Legal / sensitive
    "legal", "lawsuit", "gdpr", "data breach", "pii",
]

SYSTEM_PROMPT = """\
You are a support ticket analyzer for a multi-domain support triage system.
You will be given a support ticket and relevant documentation excerpts.

YOUR RULES:
1. Use ONLY the provided documentation to inform your analysis.
2. Do NOT use outside knowledge or make up policies.
3. If the ticket has no relation to the documentation, classify it as invalid.
4. Be conservative with risk — when in doubt, mark risk_level as HIGH.

Respond ONLY with a valid JSON object — no preamble, no markdown fences.

JSON schema:
{
  "request_type": "product_issue" | "feature_request" | "bug" | "invalid",
  "risk_level": "HIGH" | "MEDIUM" | "LOW",
  "grounded_summary": "<one sentence summary grounded in the docs, or 'No corpus coverage.'>"
}
"""


# ---------------------------------------------------------------------------
# Data types
# ---------------------------------------------------------------------------

@dataclass
class AnalysisResult:
    request_type:     str            # validated, always in ALLOWED_REQUEST_TYPES
    risk_level:       str            # validated, always in ALLOWED_RISK_LEVELS
    grounded_summary: str
    raw_response:     str = ""       # for debugging


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _format_chunks(chunks: list[dict]) -> str:
    """Format top chunks into a documentation block for the prompt."""
    if not chunks:
        return "No relevant documentation found."

    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(
            f"[Doc {i}] {chunk['article_title']} ({chunk['product_area']})\n"
            f"{chunk['text']}"
        )
    return "\n\n---\n\n".join(parts)


def _force_high_risk(issue: str) -> bool:
    """Return True if the issue contains hard-coded high-risk signals."""
    lower = issue.lower()
    return any(kw in lower for kw in HIGH_RISK_KEYWORDS)


def _parse_response(raw: str) -> tuple[str, str, str]:
    """Parse and validate the LLM JSON response.

    Returns (request_type, risk_level, grounded_summary).
    Falls back to safe defaults on any parse failure.
    """
    # Strip accidental markdown fences
    cleaned = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # Try extracting first JSON object via regex
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group())
            except json.JSONDecodeError:
                return "invalid", "HIGH", "Unable to parse LLM response."
        else:
            return "invalid", "HIGH", "Unable to parse LLM response."

    request_type = str(data.get("request_type", "invalid")).strip().lower()
    risk_level   = str(data.get("risk_level",   "HIGH")).strip().upper()
    summary      = str(data.get("grounded_summary", "")).strip()

    # Validate against allowed values — fall back to safe defaults
    if request_type not in ALLOWED_REQUEST_TYPES:
        request_type = "invalid"
    if risk_level not in ALLOWED_RISK_LEVELS:
        risk_level = "HIGH"
    if not summary:
        summary = "No grounded summary available."

    return request_type, risk_level, summary


# ---------------------------------------------------------------------------
# Main analyzer
# ---------------------------------------------------------------------------

def analyze(
    issue: str,
    top_chunks: list[dict],
    domain: Optional[str] = None,
) -> AnalysisResult:
    """
    Call the LLM to analyze a support ticket.

    Args:
        issue:      sanitized ticket text
        top_chunks: top-k retrieved corpus chunks (from classifier)
        domain:     detected domain (for prompt context, optional)
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise EnvironmentError("ANTHROPIC_API_KEY environment variable not set.")

    client = anthropic.Anthropic(api_key=api_key)

    docs_block   = _format_chunks(top_chunks)
    domain_hint  = f"Detected domain: {domain}\n" if domain else ""

    user_message = (
        f"{domain_hint}"
        f"Support ticket:\n{issue}\n\n"
        f"Relevant documentation:\n{docs_block}"
    )

    response = client.messages.create(
        model=MODEL,
        max_tokens=1000,
        temperature=0,           # deterministic
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )

    raw = response.content[0].text if response.content else ""
    request_type, risk_level, grounded_summary = _parse_response(raw)

    # Hard override: force HIGH risk if known dangerous keywords present
    if _force_high_risk(issue):
        risk_level = "HIGH"

    return AnalysisResult(
        request_type=request_type,
        risk_level=risk_level,
        grounded_summary=grounded_summary,
        raw_response=raw,
    )


# ---------------------------------------------------------------------------
# Smoke test (python analyzer.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import pickle, sys

    index_path = sys.argv[1] if len(sys.argv) > 1 else "index.pkl"
    with open(index_path, "rb") as f:
        data = pickle.load(f)
    chunks = data["chunks"]
    bm25   = data["bm25"]

    from classifier import classify

    test_cases = [
        ("How long do HackerRank tests stay active in the system?",           "HackerRank"),
        ("One of my claude conversations has private info, can I delete it?", "Claude"),
        ("My Visa card was stolen, what do I do?",                            "Visa"),
        ("What is the name of the actor in Iron Man?",                        "None"),
        ("site is completely down, nothing is accessible",                    "None"),
    ]

    print(f"{'ISSUE':<55} {'TYPE':<16} {'RISK':<7} SUMMARY")
    print("-" * 120)

    for issue, company in test_cases:
        result = classify(issue, company, chunks, bm25)
        analysis = analyze(issue, result.top_chunks, result.domain)
        print(
            f"{issue[:54]:<55} "
            f"{analysis.request_type:<16} "
            f"{analysis.risk_level:<7} "
            f"{analysis.grounded_summary[:60]}"
        )