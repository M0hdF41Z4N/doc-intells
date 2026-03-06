"""
agents/graph.py

LangGraph-based autonomous document intelligence agent.

Updated Workflow
----------------
  classify_intent (removed — subsumed into decompose+route)

  1. decompose      — QueryDecomposer splits the complex query into typed sub-queries.
  2. route          — QueryRouter builds an ordered ExecutionStep plan.
  3. execute_plan   — Iterates the plan: calls retrieve_text / retrieve_table /
                      execute_math per step, accumulating evidence across all sub-queries.
                      Self-correction: if a retrieval step returns no results the
                      LLM is only then asked to rewrite the query and retry (up to
                      2 more attempts). The LLM is NOT invoked when retrieval succeeds.
  4. synthesize     — LLM synthesises a final answer from all collected evidence.
  5. verify         — Citation validation / observability node.
  6. END

The original single-intent nodes (retrieve_text_node, retrieve_table_node,
execute_math_node) are preserved, but are now called programmatically inside
execute_plan_node rather than being wired as separate graph edges. This keeps
the graph simple while supporting arbitrarily complex multi-step plans.
"""

from __future__ import annotations

import json
import logging
from typing import Annotated, Any, Dict, List, Sequence, TypedDict

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from agents.decomposer import QueryDecomposer
from agents.router import ExecutionStep, QueryRouter
from agents.tools import MathTool, RetrievalTool, TableTool
from services.embeddings import EmbeddingsService
from services.vector_storage import VectorStorage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Self-correction constants
# ---------------------------------------------------------------------------

# How many retrieval attempts are allowed per tool call:
#   Attempt 1 — original query (LLM NOT called)
#   Attempt 2 — LLM rewrites query  (only called if attempt 1 fails)
#   Attempt 3 — LLM extracts keywords (only called if attempt 2 fails)
MAX_RETRIEVAL_ATTEMPTS = 3

# Substrings that signal the retrieval tool returned nothing useful.
# Checked with a case-insensitive substring match.
_EMPTY_RESULT_SIGNALS = {
    "no relevant text documents found",
    "no relevant tables found",
    "no results found",
    "retrieval failed",
    "table retrieval failed",
    "error: storage services not initialized",
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


class AgentState(TypedDict):
    """
    Typed state dictionary shared across all graph nodes.
    """
    messages: List[BaseMessage]
    original_query: str
    sub_queries: List[Dict[str, Any]]
    execution_plan: List[Dict[str, Any]]
    evidence: List[str]
    citations: List[Dict[str, Any]]
    final_answer: str
    trace_log: List[str]


# ---------------------------------------------------------------------------
# Pydantic output schemas (kept for compat / future use)
# ---------------------------------------------------------------------------


class IntentClassification(BaseModel):
    intent: str = Field(description="One of: 'verification', 'comparison', 'forecasting'")
    reasoning: str = Field(description="Why this intent was chosen")


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


class DocumentAgent:
    def __init__(self) -> None:
        # ---- Services ----
        self.embeddings = EmbeddingsService()
        self.vector_storage = VectorStorage(vector_size=self.embeddings.get_vector_size())
        self.vector_storage.init_collection()

        # ---- Tools ----
        self.retrieval_tool = RetrievalTool()
        self.retrieval_tool.vector_storage = self.vector_storage
        self.retrieval_tool.embeddings_service = self.embeddings

        self.table_tool = TableTool()
        self.table_tool.vector_storage = self.vector_storage
        self.table_tool.embeddings_service = self.embeddings

        self.math_tool = MathTool()

        # ---- LLMs ----
        from services.llm_factory import get_llm, get_llm_with_json

        self.llm = get_llm(temperature=0)
        self.llm_json = get_llm_with_json(self.llm)

        # ---- Decomposition & Routing Layer ----
        self.decomposer = QueryDecomposer(llm_json=self.llm_json)
        self.router = QueryRouter()

    # =========================================================================
    # Node: decompose
    # =========================================================================

    def decompose_node(self, state: AgentState) -> AgentState:
        """
        Use QueryDecomposer to break the query into typed sub-queries.
        Populates state['sub_queries'].
        """
        query = state["original_query"]
        state["trace_log"].append(f"[decompose] Breaking down query: {query!r}")

        try:
            sub_queries = self.decomposer.decompose(query)
        except Exception as exc:
            logger.error("Decomposer failed: %s", exc)
            # Graceful degradation — treat whole query as one verification sub-query
            sub_queries = [
                {"id": 1, "sub_query": query, "intent": "verification", "depends_on": []}
            ]
            state["trace_log"].append(f"[decompose] Fallback to single sub-query due to error: {exc}")

        state["sub_queries"] = sub_queries

        for sq in sub_queries:
            state["trace_log"].append(
                f"[decompose] Sub-query #{sq['id']} [{sq['intent']}]: {sq['sub_query']}"
            )

        return state

    # =========================================================================
    # Node: route
    # =========================================================================

    def route_node(self, state: AgentState) -> AgentState:
        """
        Use QueryRouter to build an ordered execution plan from the sub-queries.
        Populates state['execution_plan'] (list of dicts, serialisable).
        """
        sub_queries = state.get("sub_queries", [])
        state["trace_log"].append(f"[route] Routing {len(sub_queries)} sub-quer(ies)")

        plan: List[ExecutionStep] = self.router.route(sub_queries)
        state["execution_plan"] = [step.to_dict() for step in plan]
        state["trace_log"].append(f"[route] Plan: {self.router.summarise(plan)}")

        return state

    # =========================================================================
    # Node: execute_plan
    # =========================================================================

    def execute_plan_node(self, state: AgentState) -> AgentState:
        """
        Iterate over the execution plan and run each tool chain in order.

        For each step:
          - retrieve_text   → calls self.retrieval_tool
          - retrieve_table  → calls self.table_tool
          - execute_math    → extracts params via LLM then calls self.math_tool

        Evidence and citations are accumulated across all steps.
        """
        plan = state.get("execution_plan", [])
        state.setdefault("evidence", [])
        state.setdefault("citations", [])

        # Index already-computed evidence by sub_query_id for dependency passing
        step_results: Dict[int, str] = {}

        for step_dict in plan:
            step_id = step_dict["sub_query_id"]
            sub_query = step_dict["sub_query"]
            tool_chain = step_dict["tool_chain"]
            intent = step_dict["intent"]

            state["trace_log"].append(
                f"[execute_plan] Step #{step_id} [{intent}] '{sub_query}' → {tool_chain}"
            )

            # Enrich sub-query with dependency context if available
            dep_context = ""
            if step_dict.get("depends_on"):
                dep_snippets = [
                    step_results[d] for d in step_dict["depends_on"] if d in step_results
                ]
                if dep_snippets:
                    dep_context = "\nContext from prior steps:\n" + "\n".join(dep_snippets)

            enriched_query = sub_query + dep_context
            step_evidence_parts: List[str] = []

            for tool_name in tool_chain:
                if tool_name == "retrieve_text":
                    # Attempt retrieval; LLM rewrite fires ONLY if result is empty
                    result = self._self_correct_retrieval(
                        query=enriched_query,
                        tool_fn=self._run_retrieval,
                        tool_label="retrieve_text",
                        state=state,
                    )
                    step_evidence_parts.append(f"Text Evidence:\n{result}")
                    self._parse_citations(result, state)

                elif tool_name == "retrieve_table":
                    # Attempt retrieval; LLM rewrite fires ONLY if result is empty
                    result = self._self_correct_retrieval(
                        query=enriched_query,
                        tool_fn=self._run_table,
                        tool_label="retrieve_table",
                        state=state,
                    )
                    step_evidence_parts.append(f"Table Evidence:\n{result}")

                elif tool_name == "execute_math":
                    # Math needs the evidence gathered so far in this step
                    accumulated_evidence_for_math = "\n".join(step_evidence_parts)
                    result = self._run_math(sub_query, accumulated_evidence_for_math, state)
                    step_evidence_parts.append(f"Math Result:\n{result}")

                else:
                    state["trace_log"].append(f"[execute_plan] Unknown tool '{tool_name}' — skipping.")

            # Store aggregated evidence for this step
            combined = "\n".join(step_evidence_parts)
            step_results[step_id] = combined
            state["evidence"].append(f"--- Sub-query #{step_id}: {sub_query} ---\n{combined}")

        return state

    # =========================================================================
    # Node: synthesize
    # =========================================================================

    def synthesize_answer(self, state: AgentState) -> AgentState:
        state["trace_log"].append("[synthesize] Synthesising final answer")

        evidence_text = "\n\n".join(state.get("evidence", []))

        prompt = f"""User Query: {state["original_query"]}

Evidence Gathered (from multiple retrieval & computation steps):
{evidence_text}

You are a factual, autonomous document intelligence analyst.
Answer the user query strictly based on the provided evidence.
If the evidence does not contain the answer, say "I cannot find the answer in the provided documents."
Be concise, accurate, and incorporate exact numbers or citations where present.
"""
        result = self.llm.invoke([SystemMessage(content=prompt)])
        state["final_answer"] = result.content
        return state

    # =========================================================================
    # Node: verify
    # =========================================================================

    def verify_citations(self, state: AgentState) -> AgentState:
        """Observability node — validates citations exist and logs execution summary."""
        state["trace_log"].append("[verify] Verifying citations & finalising trace")

        if not state.get("citations"):
            state["trace_log"].append(
                "[verify] Warning: No page-level citations extracted. "
                "Answer may lack verifiable evidence."
            )
        else:
            state["trace_log"].append(
                f"[verify] {len(state['citations'])} citation(s) verified."
            )

        return state

    # =========================================================================
    # Build graph
    # =========================================================================

    def build(self) -> StateGraph:
        workflow = StateGraph(AgentState)

        # Nodes
        workflow.add_node("decompose", self.decompose_node)
        workflow.add_node("route", self.route_node)
        workflow.add_node("execute_plan", self.execute_plan_node)
        workflow.add_node("synthesize", self.synthesize_answer)
        workflow.add_node("verify", self.verify_citations)

        # Edges — linear pipeline; branching is handled inside execute_plan_node
        workflow.set_entry_point("decompose")
        workflow.add_edge("decompose", "route")
        workflow.add_edge("route", "execute_plan")
        workflow.add_edge("execute_plan", "synthesize")
        workflow.add_edge("synthesize", "verify")
        workflow.add_edge("verify", END)

        return workflow.compile()

    # =========================================================================
    # Private tool runners
    # =========================================================================

    def _run_retrieval(self, query: str, state: AgentState) -> str:
        state["trace_log"].append(f"[tool] retrieve_text ← '{query[:80]}'")
        return self.retrieval_tool._run(query)

    def _run_table(self, query: str, state: AgentState) -> str:
        state["trace_log"].append(f"[tool] retrieve_table ← '{query[:80]}'")
        return self.table_tool._run(query)

    def _run_math(self, query: str, evidence: str, state: AgentState) -> str:
        """
        Ask the LLM to extract math parameters from evidence, then call MathTool.
        """
        state["trace_log"].append("[tool] execute_math — extracting parameters")

        prompt = f"""Based on the User Query: "{query}" and Evidence: "{evidence}",
determine the exact inputs to a calculator for a CAGR or percentage calculation.

Return JSON EXACTLY matching:
{{"operation": "cagr|percentage|average", "values": [num1, num2], "years": float_or_null}}

If you cannot determine the values, return empty lists."""

        extract_res = self.llm_json.invoke([HumanMessage(content=prompt)])
        try:
            params = json.loads(extract_res.content)
            operation = params.get("operation")
            values = params.get("values", [])
            years = params.get("years")

            if operation and values:
                state["trace_log"].append(
                    f"[tool] execute_math params: op={operation} vals={values} yrs={years}"
                )
                return self.math_tool._run(operation=operation, values=values, years=years)
            else:
                state["trace_log"].append("[tool] execute_math — insufficient parameters extracted.")
                return "Math calculation skipped: could not extract parameters from evidence."

        except (json.JSONDecodeError, Exception) as exc:
            state["trace_log"].append(f"[tool] execute_math — parameter extraction failed: {exc}")
            return f"Math calculation failed: {exc}"

    def _parse_citations(self, result: str, state: AgentState) -> None:
        """Extract page citations from formatted retrieval output."""
        for line in result.split("\n\n"):
            if line.startswith("[Page"):
                try:
                    page_part, text_part = line.split("]: ", 1)
                    page_num = page_part.replace("[Page ", "")
                    citation = {"page": page_num, "text": text_part[:150] + "..."}
                    if citation not in state["citations"]:
                        state["citations"].append(citation)
                except ValueError:
                    continue

    # =========================================================================
    # Self-correction: retry retrieval with LLM query rewriting
    # =========================================================================

    def _is_empty_result(self, result: str) -> bool:
        """
        Return True only when *result* contains a known empty-retrieval sentinel.
        The check is intentionally narrowly scoped so that short-but-valid
        responses from the tool are never mistakenly treated as failures.
        """
        lowered = result.lower()
        return any(signal in lowered for signal in _EMPTY_RESULT_SIGNALS)

    def _self_correct_retrieval(
        self,
        query: str,
        tool_fn,
        tool_label: str,
        state: AgentState,
    ) -> str:
        """
        Run *tool_fn* and self-correct only if the result is empty.

        Attempt 1  — call tool with the original query.
                     If it returns results → return immediately, LLM never called.

        Attempt 2  — ONLY if attempt 1 was empty:
                     ask the LLM to rephrase the query with different vocabulary,
                     then retry the tool.

        Attempt 3  — ONLY if attempt 2 was also empty:
                     ask the LLM to extract bare keywords (broadest possible search),
                     then retry the tool.

        If all three attempts fail the tool's last response is returned so that
        the synthesise node can tell the user no evidence was found.
        """
        # ------------------------------------------------------------------ #
        # Attempt 1 — original query, NO LLM call                            #
        # ------------------------------------------------------------------ #
        state["trace_log"].append(
            f"[self-correct/{tool_label}] Attempt 1/{MAX_RETRIEVAL_ATTEMPTS} "
            f"— original query: '{query[:80]}'"
        )
        result = tool_fn(query, state)

        if not self._is_empty_result(result):
            # Success on first try — LLM was never involved
            return result

        state["trace_log"].append(
            f"[self-correct/{tool_label}] Attempt 1 returned empty "
            "— invoking LLM to rewrite query."
        )

        # ------------------------------------------------------------------ #
        # Attempt 2 — LLM rewrites the query                                 #
        # ------------------------------------------------------------------ #
        rewritten = self._rewrite_query(query, state, tool_label)
        state["trace_log"].append(
            f"[self-correct/{tool_label}] Attempt 2/{MAX_RETRIEVAL_ATTEMPTS} "
            f"— rewritten query: '{rewritten[:80]}'"
        )
        result = tool_fn(rewritten, state)

        if not self._is_empty_result(result):
            state["trace_log"].append(
                f"[self-correct/{tool_label}] Attempt 2 succeeded."
            )
            return result

        state["trace_log"].append(
            f"[self-correct/{tool_label}] Attempt 2 returned empty "
            "— invoking LLM for keyword broadening."
        )

        # ------------------------------------------------------------------ #
        # Attempt 3 — LLM extracts keywords (widest recall)                  #
        # ------------------------------------------------------------------ #
        keywords = self._extract_keywords(query, state, tool_label)
        state["trace_log"].append(
            f"[self-correct/{tool_label}] Attempt 3/{MAX_RETRIEVAL_ATTEMPTS} "
            f"— keyword query: '{keywords[:80]}'"
        )
        result = tool_fn(keywords, state)

        if self._is_empty_result(result):
            state["trace_log"].append(
                f"[self-correct/{tool_label}] All {MAX_RETRIEVAL_ATTEMPTS} attempts "
                "yielded no results. Evidence gap logged."
            )
        else:
            state["trace_log"].append(
                f"[self-correct/{tool_label}] Attempt 3 succeeded via keyword broadening."
            )

        return result

    def _rewrite_query(self, query: str, state: AgentState, tool_label: str) -> str:
        """
        Ask the LLM to rephrase *query* using different vocabulary.

        Called ONLY when the first retrieval attempt failed.
        Falls back silently to the original query if the LLM call itself fails,
        ensuring the retry still happens without crashing.
        """
        prompt = (
            "You are a query rewriting assistant for a document retrieval system.\n"
            "The following question produced no results when searched against a vector database.\n"
            "Rewrite it using different vocabulary and phrasing to maximise the chance of "
            "finding relevant passages.\n"
            "Return ONLY the rewritten question — no explanation, no quotes.\n\n"
            f"Failed query: {query}"
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            rewritten = response.content.strip().strip('"').strip("'")
            if not rewritten or len(rewritten) < 5:
                raise ValueError("Rewritten query too short.")
            return rewritten
        except Exception as exc:
            logger.warning(
                "[self-correct/%s] Query rewriting LLM call failed (%s); using original.",
                tool_label, exc,
            )
            state["trace_log"].append(
                f"[self-correct/{tool_label}] Query rewriting failed ({exc}); retrying with original."
            )
            return query

    def _extract_keywords(self, query: str, state: AgentState, tool_label: str) -> str:
        """
        Ask the LLM to distil *query* into its most important search keywords.

        Called ONLY when both the original and rewritten queries failed.
        Falls back to a simple stopword-stripped version of the query without
        crashing if the LLM call fails.
        """
        prompt = (
            "You are a keyword extraction assistant.\n"
            "Both the original question and a rewritten version failed to retrieve "
            "any documents from a vector database.\n"
            "Extract the 3-6 most important search keywords from the question below.\n"
            "Return ONLY a space-separated list of keywords — no punctuation, no explanation.\n\n"
            f"Question: {query}"
        )
        try:
            response = self.llm.invoke([HumanMessage(content=prompt)])
            keywords = response.content.strip()
            if not keywords or len(keywords) < 3:
                raise ValueError("Keyword extraction returned too little.")
            return keywords
        except Exception as exc:
            logger.warning(
                "[self-correct/%s] Keyword extraction LLM call failed (%s); using heuristic fallback.",
                tool_label, exc,
            )
            state["trace_log"].append(
                f"[self-correct/{tool_label}] Keyword extraction failed ({exc}); using heuristic."
            )
            # Heuristic fallback: strip common stopwords and punctuation
            _STOPWORDS = {
                "what", "is", "the", "a", "an", "of", "to", "in", "for", "and",
                "or", "how", "are", "was", "were", "did", "does", "do", "from",
                "with", "that", "this", "it", "be", "been", "by", "at", "on",
            }
            tokens = [
                t.strip("?.,!;:") for t in query.lower().split()
                if t.strip("?.,!;:") not in _STOPWORDS and len(t.strip("?.,!;:")) > 2
            ]
            return " ".join(tokens) if tokens else query
