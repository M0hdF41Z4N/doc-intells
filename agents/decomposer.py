"""
agents/decomposer.py

QueryDecomposer — Breaks a complex user query into a list of atomic sub-queries,
each annotated with its intent type so the router can select the right tool chain.

Sub-query schema:
  {
    "id":          int,          # 1-based execution order
    "sub_query":   str,          # The atomised question
    "intent":      str,          # "verification" | "comparison" | "forecasting"
    "depends_on":  List[int]     # IDs of sub-queries whose result this one needs
  }

Design notes:
  - A simple query with a single, clear intent returns exactly one sub-query so the
    overhead is zero in the happy path.
  - Decomposition uses JSON-mode (or prompt-engineering fallback) to guarantee
    parseable output regardless of LLM provider.
  - The class is stateless; each call to `decompose` is independent.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Dict, List

from langchain_core.messages import HumanMessage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Schema helpers
# ---------------------------------------------------------------------------

VALID_INTENTS = {"verification", "comparison", "forecasting"}

_SYSTEM_PROMPT = """You are a query decomposition engine for a document intelligence system.
Your job is to analyse the user's question and break it into the minimum number of
atomic sub-questions needed to fully answer it.

Rules:
1. If the query is already simple and atomic, return exactly ONE sub-query.
2. Assign one of three intent labels to every sub-query:
   - "verification"  — looking up a specific fact, number, or claim from the text.
   - "comparison"    — comparing two or more data points (usually needs table data).
   - "forecasting"   — requires a mathematical calculation (CAGR, projection, etc.).
3. Record dependencies: if sub-query 3 needs the result of sub-query 1, set
   depends_on=[1] for sub-query 3.
4. Keep sub-queries independent where possible.
5. Output ONLY valid JSON — no preamble, no markdown fences.

Output format (array, even for a single sub-query):
[
  {
    "id": 1,
    "sub_query": "<atomic question>",
    "intent": "verification|comparison|forecasting",
    "depends_on": []
  },
  ...
]"""


def _clean_json(raw: str) -> str:
    """Strip markdown code fences if the LLM wrapped the JSON anyway."""
    raw = raw.strip()
    if raw.startswith("```"):
        lines = raw.splitlines()
        # drop opening fence (```json or ```) and closing fence
        lines = [l for l in lines if not l.strip().startswith("```")]
        raw = "\n".join(lines).strip()
    return raw


def _validate_sub_query(item: Any) -> Dict[str, Any]:
    """Ensure a parsed dict conforms to the expected schema, filling defaults."""
    if not isinstance(item, dict):
        raise ValueError(f"Sub-query is not a dict: {item!r}")

    intent = str(item.get("intent", "verification")).lower()
    if intent not in VALID_INTENTS:
        logger.warning("Unknown intent '%s'; defaulting to 'verification'.", intent)
        intent = "verification"

    return {
        "id": int(item.get("id", 1)),
        "sub_query": str(item.get("sub_query", "")).strip(),
        "intent": intent,
        "depends_on": [int(d) for d in item.get("depends_on", [])],
    }


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class QueryDecomposer:
    """
    Decomposes a natural-language query into an ordered list of typed sub-queries.

    Parameters
    ----------
    llm_json :
        A LangChain chat-model instance configured for JSON output.
        Obtain one via ``services.llm_factory.get_llm_with_json(get_llm())``.
    """

    def __init__(self, llm_json) -> None:
        self._llm = llm_json

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def decompose(self, query: str) -> List[Dict[str, Any]]:
        """
        Break *query* into atomic sub-queries.

        Returns
        -------
        List[Dict]
            Ordered list of sub-query dicts, each with keys:
            ``id``, ``sub_query``, ``intent``, ``depends_on``.

        Raises
        ------
        ValueError
            If the LLM returns output that cannot be parsed into the expected
            schema even after sanitisation.
        """
        query = query.strip()
        if not query:
            raise ValueError("Cannot decompose an empty query.")

        logger.info("Decomposing query: %r", query)

        user_message = f"Decompose this query:\n\n{query}"
        response = self._llm.invoke(
            [
                HumanMessage(content=f"{_SYSTEM_PROMPT}\n\n{user_message}")
            ]
        )

        raw_content = response.content if hasattr(response, "content") else str(response)
        cleaned = _clean_json(raw_content)

        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse decomposer output: %s\nRaw: %s", exc, cleaned)
            # Graceful fallback: treat the whole query as a single verification sub-query
            logger.warning("Falling back to single-sub-query decomposition.")
            return self._fallback(query)

        if not isinstance(parsed, list):
            # Some models wrap the array in an object key
            if isinstance(parsed, dict):
                for candidate_key in ("sub_queries", "subqueries", "queries", "results"):
                    if candidate_key in parsed and isinstance(parsed[candidate_key], list):
                        parsed = parsed[candidate_key]
                        break
                else:
                    logger.warning("Unexpected JSON shape; falling back.")
                    return self._fallback(query)
            else:
                logger.warning("Unexpected JSON type; falling back.")
                return self._fallback(query)

        if not parsed:
            return self._fallback(query)

        sub_queries = []
        for item in parsed:
            try:
                sub_queries.append(_validate_sub_query(item))
            except (ValueError, TypeError) as exc:
                logger.warning("Skipping malformed sub-query %r: %s", item, exc)

        if not sub_queries:
            return self._fallback(query)

        # Re-index to guarantee monotonically increasing IDs starting from 1
        for idx, sq in enumerate(sub_queries, start=1):
            sq["id"] = idx

        logger.info("Decomposed into %d sub-quer(ies): %s", len(sub_queries), [s["sub_query"] for s in sub_queries])
        return sub_queries

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _fallback(query: str) -> List[Dict[str, Any]]:
        """Return a single-element list treating the raw query as verification."""
        return [
            {
                "id": 1,
                "sub_query": query,
                "intent": "verification",
                "depends_on": [],
            }
        ]
