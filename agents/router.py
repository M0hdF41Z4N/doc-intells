"""
agents/router.py

QueryRouter — Takes the list of typed sub-queries produced by QueryDecomposer
and returns a concrete, ordered execution plan.

Each step in the plan tells the graph exactly which nodes to visit
for that sub-query:

  ExecutionStep schema:
  {
    "sub_query_id":  int,
    "sub_query":     str,
    "intent":        str,
    "depends_on":    List[int],
    "tool_chain":    List[str]   # ordered list of node names to execute
  }

Tool-chain rules (deterministic — no LLM involved):
  verification  → ["retrieve_text"]
  comparison    → ["retrieve_table"]
  forecasting   → ["retrieve_text", "execute_math"]

The router also validates dependency ordering so the graph can execute
dependent sub-queries after their prerequisites are fulfilled.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Routing table — maps intent → ordered list of graph node names
# ---------------------------------------------------------------------------

INTENT_TOOL_CHAINS: Dict[str, List[str]] = {
    "verification": ["retrieve_text"],
    "comparison": ["retrieve_table"],
    "forecasting": ["retrieve_text", "execute_math"],
}

DEFAULT_CHAIN = ["retrieve_text"]


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


class ExecutionStep:
    """
    Represents a single executable unit in the overall plan.

    Attributes
    ----------
    sub_query_id : int
    sub_query : str
    intent : str
    depends_on : List[int]
    tool_chain : List[str]
        e.g. ["retrieve_text", "execute_math"]
    """

    __slots__ = ("sub_query_id", "sub_query", "intent", "depends_on", "tool_chain")

    def __init__(
        self,
        sub_query_id: int,
        sub_query: str,
        intent: str,
        depends_on: List[int],
        tool_chain: List[str],
    ) -> None:
        self.sub_query_id = sub_query_id
        self.sub_query = sub_query
        self.intent = intent
        self.depends_on = depends_on
        self.tool_chain = tool_chain

    def to_dict(self) -> Dict[str, Any]:
        return {
            "sub_query_id": self.sub_query_id,
            "sub_query": self.sub_query,
            "intent": self.intent,
            "depends_on": self.depends_on,
            "tool_chain": self.tool_chain,
        }

    def __repr__(self) -> str:  # pragma: no cover
        return (
            f"ExecutionStep(id={self.sub_query_id}, intent={self.intent!r}, "
            f"chain={self.tool_chain}, deps={self.depends_on})"
        )


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class QueryRouter:
    """
    Converts a list of decomposed sub-queries into an ordered execution plan.

    The routing logic is fully deterministic and does not call any LLM.

    Usage
    -----
    ::

        router = QueryRouter()
        plan = router.route(sub_queries)   # sub_queries from QueryDecomposer
        for step in plan:
            print(step.tool_chain)
    """

    def route(self, sub_queries: List[Dict[str, Any]]) -> List[ExecutionStep]:
        """
        Produce an ordered list of ExecutionStep objects.

        Parameters
        ----------
        sub_queries :
            Output of ``QueryDecomposer.decompose()``.

        Returns
        -------
        List[ExecutionStep]
            Topologically sorted (dependencies first) execution plan.
        """
        if not sub_queries:
            logger.warning("Router received an empty sub-query list.")
            return []

        # Build a lookup for dependency resolution
        id_to_sq: Dict[int, Dict[str, Any]] = {sq["id"]: sq for sq in sub_queries}

        steps: List[ExecutionStep] = []
        for sq in sub_queries:
            intent = sq.get("intent", "verification")
            tool_chain = INTENT_TOOL_CHAINS.get(intent, DEFAULT_CHAIN)

            step = ExecutionStep(
                sub_query_id=sq["id"],
                sub_query=sq["sub_query"],
                intent=intent,
                depends_on=sq.get("depends_on", []),
                tool_chain=list(tool_chain),  # defensive copy
            )
            steps.append(step)

            logger.debug(
                "Routed sub-query #%d ('%s') → %s (deps=%s)",
                step.sub_query_id,
                step.sub_query[:60],
                step.tool_chain,
                step.depends_on,
            )

        # Topological sort so dependent steps always come after their prerequisites
        steps = self._topological_sort(steps)

        logger.info(
            "Execution plan (%d step(s)): %s",
            len(steps),
            [(s.sub_query_id, s.tool_chain) for s in steps],
        )
        return steps

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _topological_sort(steps: List[ExecutionStep]) -> List[ExecutionStep]:
        """
        Return *steps* in an order where every step appears after all its
        dependencies.  Uses Kahn's algorithm (BFS).

        Falls back to the original order if a cycle is detected (which
        indicates a malformed decomposition — logged as a warning).
        """
        id_to_step = {s.sub_query_id: s for s in steps}
        in_degree: Dict[int, int] = {s.sub_query_id: 0 for s in steps}
        dependents: Dict[int, List[int]] = {s.sub_query_id: [] for s in steps}

        for step in steps:
            for dep_id in step.depends_on:
                if dep_id in in_degree:
                    in_degree[step.sub_query_id] += 1
                    dependents[dep_id].append(step.sub_query_id)
                else:
                    logger.warning(
                        "Sub-query #%d depends on unknown id #%d — ignoring dependency.",
                        step.sub_query_id,
                        dep_id,
                    )

        # BFS queue: start with all steps that have no dependencies
        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        queue.sort()  # deterministic ordering for independent steps
        sorted_steps: List[ExecutionStep] = []

        while queue:
            current_id = queue.pop(0)
            sorted_steps.append(id_to_step[current_id])
            for dependent_id in sorted(dependents[current_id]):
                in_degree[dependent_id] -= 1
                if in_degree[dependent_id] == 0:
                    queue.append(dependent_id)

        if len(sorted_steps) != len(steps):
            logger.warning(
                "Cycle detected in sub-query dependency graph. "
                "Using original order as fallback."
            )
            return steps

        return sorted_steps

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    @staticmethod
    def summarise(plan: List[ExecutionStep]) -> str:
        """Return a human-readable one-liner summary of the plan."""
        if not plan:
            return "Empty plan."
        parts = []
        for step in plan:
            chain_str = " → ".join(step.tool_chain)
            dep_str = f" (after #{step.depends_on})" if step.depends_on else ""
            parts.append(f"[#{step.sub_query_id}: {step.intent}] {chain_str}{dep_str}")
        return " | ".join(parts)
