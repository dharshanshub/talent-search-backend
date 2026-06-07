from __future__ import annotations

import json
from typing import AsyncGenerator

import structlog
from openai import AsyncOpenAI, APIError, APITimeoutError, RateLimitError

from app.core.exceptions import UpstreamServiceError
from app.schemas.search import CandidateMatch, ConversationMessage
from app.services.search_service import SearchService

logger = structlog.get_logger(__name__)

# ── System prompt ─────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are TalentAI, an expert talent acquisition assistant with deep expertise in technical hiring, recruitment strategy, and candidate evaluation. You help recruiters and hiring managers discover, analyze, and compare candidates from a proprietary talent database.

## Your Capabilities
- Search a semantic vector database of candidate profiles using natural language
- Provide insightful analysis of candidate fit and differentiators
- Compare candidates objectively in structured markdown tables
- Answer follow-up questions using full conversation context
- Maintain context across up to 30 conversation turns

## Tool: search_candidates
You have one tool: `search_candidates(query: str)` — performs semantic search over a vector database of resumes and returns ranked candidate profiles.

### CALL the tool when the user:
- Asks to find, search, discover, or show candidates
- Describes requirements for a role, technology stack, or experience level
- Mentions specific skills, seniority, location, or domain (fintech, healthtech, etc.)
- Says "find me", "show me", "search for", "who has", "I need someone who"
- Wants a recommendation or suggestions for a position
- Asks for someone "similar to" a candidate but with different criteria

### DO NOT call the tool when:
- The message is conversational or social (hi, hello, thanks, how are you, great, perfect)
- User asks to compare, rank, analyze, or summarize candidates already returned in this conversation — use the conversation context instead
- User asks a follow-up about a specific candidate shown earlier in this conversation
- User asks clarifying questions or general questions about your capabilities
- User asks you to explain or elaborate on data already in context

## Response Standards

### After a successful search
Write 3-5 sentences that provide genuine recruiter value:
1. Overall quality assessment — how well does the result set match the query?
2. Call out the top 1-2 candidates by name and explain specifically WHY they stand out (skills, experience depth, domain fit)
3. Highlight key differentiators across the set (e.g., "Most have Python but only two have production ML experience")
4. Note any trade-offs or gaps the recruiter should be aware of

### When comparing candidates — ALWAYS use a markdown table
Structure the comparison with relevant columns. Always include:

| Candidate | Role | Experience | Key Skills | Domain | Top Strength | Best Fit For |
|-----------|------|-----------|------------|--------|-------------|-------------|
| Name      | ...  | X yrs     | ...        | ...    | ...         | ...         |

After the table, add a 2-3 sentence **Recommendation** explaining the best pick given typical hiring context.

### For follow-up questions about shown candidates
- Reference candidates by name
- Quote specific details from the profiles (years, skills, location, score)
- Never say "I don't have that information" if the data was already returned earlier in the conversation

### When search returns no results
Explain clearly what criteria returned nothing, then suggest 2-3 concrete ways to broaden the search.

### For conversational messages
Respond naturally and concisely. For greetings, introduce yourself briefly and invite the user to search.

---

## Few-Shot Examples

**Greeting**
User: "hi"
Assistant: "Hello! I'm TalentAI, your AI-powered talent search assistant. I can help you find and evaluate candidates from our talent pool using semantic search and AI analysis. What kind of role or skill set are you looking for today?"

**Simple search**
User: "find me a senior Python engineer with ML experience in London"
[calls search_candidates("senior Python engineer machine learning London")]
Assistant: "I found 5 strong candidates, all London-based. Sarah Chen is the standout — 9 years of Python with production ML systems deployed at scale in a healthtech company, and she's active in the open-source ML community. Marcus Webb brings exceptional MLOps depth (Kubeflow, Airflow, Spark) making him ideal if your bottleneck is pipeline reliability rather than model research. The remaining three are solid mid-senior engineers with Python and data science backgrounds, though lighter on ML production experience. Here are all the profiles:"

**Comparison request (no tool call — use conversation context)**
User: "compare Sarah and Marcus side by side"
Assistant: "Here's a head-to-head comparison:

| | Sarah Chen | Marcus Webb |
|---|---|---|
| Experience | 9 years | 7 years |
| Seniority | Staff | Senior |
| Core Strength | Deep ML research + production systems | MLOps & pipeline architecture |
| Key Skills | Python, PyTorch, MLflow, FastAPI | Python, Kubeflow, Airflow, Spark |
| Domain | Healthtech | Fintech |
| Location | London | London |
| Match Score | 94% | 87% |
| Best For | ML-heavy product teams | Data platforms, MLOps infrastructure |

**Recommendation:** Choose Sarah if you need someone to own model development end-to-end — her research background plus production experience is rare. Choose Marcus if your immediate challenge is data infrastructure reliability and pipeline orchestration before scaling model work."

**Follow-up question (no tool call)**
User: "which one is better for a Series A startup?"
Assistant: "For a Series A startup, Marcus is typically the stronger choice. His MLOps background means he can architect the entire data infrastructure — pipelines, orchestration, monitoring — from scratch with limited resources. Sarah's deeper specialization in model research becomes more valuable once you have a stable data platform and are ready to optimize models at scale. That said, if your Series A product is ML-native (recommendations, NLP, CV), Sarah's ability to ship production models quickly could be the differentiator you need."

**Refined search (calls tool again)**
User: "show me someone similar to Marcus but based in Berlin"
[calls search_candidates("senior MLOps data engineer Kubeflow Airflow pipeline Berlin")]
"""

# ── Tool definition ────────────────────────────────────────────────────────────

_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_candidates",
            "description": (
                "Searches the talent database for candidate profiles that match the described requirements. "
                "Call this whenever the user wants to find, discover, or get recommendations for candidates. "
                "Construct the query to include role, skills, seniority, location, and domain if mentioned."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": (
                            "A detailed natural-language description of the ideal candidate. "
                            "Include: role/title, required skills, seniority level, location (if specified), "
                            "domain or industry (if specified), and years of experience (if specified). "
                            "Example: 'senior Python engineer machine learning PyTorch London fintech 7+ years'"
                        ),
                    }
                },
                "required": ["query"],
            },
        },
    }
]


# ── Agent service ─────────────────────────────────────────────────────────────

class AgentService:
    """Conversational agent that routes between direct answers and vector search.

    Uses OpenAI tool calling to decide when to query Pinecone. Streams the final
    answer back to the caller as Server-Sent Events (SSE).
    """

    # Keep the last N conversation turns to stay within context limits
    _MAX_HISTORY = 30

    def __init__(
        self,
        openai_client: AsyncOpenAI,
        search_service: SearchService,
        model: str,
    ) -> None:
        self._client = openai_client
        self._search = search_service
        self._model = model

    async def run_stream(
        self,
        query: str,
        history: list[ConversationMessage],
        top_k: int | None,
        request_id: str,
    ) -> AsyncGenerator[str, None]:
        """Run the agent and yield SSE-formatted event strings.

        Event types:
          {"type": "candidates", "data": [...]}   — emitted before streaming if search ran
          {"type": "delta",      "content": "..."}— streamed text chunk
          {"type": "done"}                         — stream complete
          {"type": "error",      "message": "..."}— terminal error
        """
        # Build the messages array: system + trimmed history + current user turn
        messages: list[dict] = [{"role": "system", "content": _SYSTEM_PROMPT}]
        for msg in history[-self._MAX_HISTORY:]:
            messages.append({"role": msg.role, "content": msg.content})
        messages.append({"role": "user", "content": query})

        # ── Phase 1: tool-calling decision (non-streaming, fast) ──────────────
        try:
            decision = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                tools=_TOOLS,
                tool_choice="auto",
                temperature=0.2,
            )
        except (APIError, APITimeoutError, RateLimitError) as exc:
            logger.error("agent_decision_failed", error=str(exc), request_id=request_id)
            yield _sse_error("I'm having trouble connecting to the AI service. Please try again.")
            return
        except Exception as exc:
            logger.error("agent_decision_unexpected", error=str(exc), request_id=request_id)
            yield _sse_error("Unexpected error. Please try again.")
            return

        assistant_msg = decision.choices[0].message
        candidates: list[CandidateMatch] = []

        # ── Phase 2: execute tool if requested ────────────────────────────────
        if assistant_msg.tool_calls:
            tool_call = assistant_msg.tool_calls[0]
            try:
                args = json.loads(tool_call.function.arguments)
            except json.JSONDecodeError:
                args = {}

            search_query = args.get("query", query)
            logger.info(
                "agent_tool_call",
                tool="search_candidates",
                search_query=search_query[:200],
                request_id=request_id,
            )

            try:
                candidates = await self._search.retrieve(search_query, top_k, request_id)
            except UpstreamServiceError as exc:
                logger.error("agent_retrieve_failed", error=str(exc), request_id=request_id)
                # Continue without candidates — agent will say nothing was found
                candidates = []
            except Exception as exc:
                logger.error("agent_retrieve_unexpected", error=str(exc), request_id=request_id)
                candidates = []

            # Emit candidates immediately so the UI can render cards while the
            # text answer is still being streamed
            if candidates:
                yield _sse_event("candidates", {"data": [_candidate_json(c) for c in candidates]})

            # Append tool call + result to messages for final answer generation
            messages.append({
                "role": "assistant",
                "tool_calls": [
                    {
                        "id": tool_call.id,
                        "type": "function",
                        "function": {
                            "name": tool_call.function.name,
                            "arguments": tool_call.function.arguments,
                        },
                    }
                ],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": _format_candidates_for_llm(candidates, search_query),
            })

        # ── Phase 3: stream the final answer ──────────────────────────────────
        try:
            stream = await self._client.chat.completions.create(
                model=self._model,
                messages=messages,
                temperature=0.4,
                stream=True,
                max_tokens=1024,
            )

            async for chunk in stream:
                delta = chunk.choices[0].delta.content
                if delta:
                    yield _sse_event("delta", {"content": delta})

        except (APIError, APITimeoutError, RateLimitError) as exc:
            logger.error("agent_stream_failed", error=str(exc), request_id=request_id)
            yield _sse_error("Streaming failed. Please try again.")
            return
        except Exception as exc:
            logger.error("agent_stream_unexpected", error=str(exc), request_id=request_id)
            yield _sse_error("Unexpected streaming error.")
            return

        yield _sse_event("done", {})
        logger.info(
            "agent_done",
            tool_used=bool(assistant_msg.tool_calls),
            candidates_found=len(candidates),
            request_id=request_id,
        )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _sse_event(event_type: str, payload: dict) -> str:
    data = {"type": event_type, **payload}
    return f"data: {json.dumps(data)}\n\n"


def _sse_error(message: str) -> str:
    return _sse_event("error", {"message": message})


def _candidate_json(c: CandidateMatch) -> dict:
    return c.model_dump(mode="json")


def _format_candidates_for_llm(candidates: list[CandidateMatch], query: str) -> str:
    """Format retrieved candidates as structured text for the LLM context window."""
    if not candidates:
        return f"No candidates found matching: {query}. Inform the user and suggest they broaden their search criteria."

    lines = [f"Search query: '{query}'\nFound {len(candidates)} matching candidates:\n"]
    for i, c in enumerate(candidates, 1):
        lines.append(
            f"{i}. {c.name}\n"
            f"   Title: {c.title}\n"
            f"   Location: {c.location}\n"
            f"   Experience: {c.years_experience} years\n"
            f"   Skills: {', '.join(c.skills[:12])}\n"
            f"   Match score: {c.score:.0%}\n"
            f"   Last updated: {c.last_updated}\n"
            f"   Summary: {c.summary or 'Not available'}\n"
        )

    lines.append(
        "\nProvide an insightful recruiter-facing summary following the response standards in your instructions. "
        "Mention the top candidates by name, explain WHY they are strong matches, and highlight key differentiators."
    )
    return "\n".join(lines)
