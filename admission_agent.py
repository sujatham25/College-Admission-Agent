"""
admission_agent.py
------------------
Core logic of the College Admission Agent.

Flow
----
1. Receive student question.
2. Retrieve top-k relevant chunks from the knowledge base (RAG).
3. If no relevant chunks found → return "not available" response without LLM.
4. Build a grounded prompt with retrieved context.
5. Call IBM Granite via watsonx.ai to generate the final answer.
6. Return the answer along with source references.

⚠️  DEMO PROJECT — IBM AICTE Internship 2026
"""

from __future__ import annotations

import textwrap
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import config
from rag.retriever import Chunk, retrieve


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class RetrievedContext:
    """A single retrieved context item shown to the student as a source."""
    source: str
    excerpt: str
    score: float


@dataclass
class AgentResponse:
    """The full response returned by the admission agent."""
    answer: str
    sources: List[RetrievedContext] = field(default_factory=list)
    is_fallback: bool = False         # True when KB had no relevant info
    error: Optional[str] = None       # populated if the LLM call failed


# ---------------------------------------------------------------------------
# Prompt builder
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = """\
You are a helpful College Admission Assistant for Demo College of Technology.
Your job is to answer students' questions about college admissions accurately \
and clearly, using ONLY the information provided in the CONTEXT below.

Rules you MUST follow:
1. Answer ONLY based on the CONTEXT. Do not add information that is not in the CONTEXT.
2. If the CONTEXT does not contain the answer, say exactly: \
"I'm sorry, I could not find this information in the admission knowledge base. \
Please contact the admissions office directly."
3. Never guess or make up fee amounts, eligibility criteria, deadlines, or document requirements.
4. Be concise, friendly, and professional.
5. If the CONTEXT mentions that data is sample/demo data, you may mention that to the student.
5. Treat the CONTEXT as the authoritative knowledge base for this demo project. Do not reject or ignore information merely because it is labeled sample/demo data.
6. When multiple sources are provided, prefer the source that directly matches the student's question and ignore unrelated information.
7. For eligibility questions, report all relevant eligibility requirements found in the context, including subjects, minimum marks, age requirements, and entrance-exam requirements.
8. Do not combine contradictory statements. If sources conflict, state that the information is conflicting and do not guess.
"""

def _build_prompt(question: str, chunks: List[Tuple[Chunk, float]]) -> str:
    """
    Construct the full prompt sent to IBM Granite.

    The retrieved chunks form the CONTEXT block. The model is instructed to
    answer only from this context.
    """
    context_parts: List[str] = []
    for i, (chunk, score) in enumerate(chunks, start=1):
        context_parts.append(
            f"[Source {i}: {chunk.source}]\n{chunk.text}"
        )
    context_block = "\n\n---\n\n".join(context_parts)

    prompt = f"""{SYSTEM_PROMPT}

CONTEXT:
{context_block}

---

Student Question: {question}

Answer:"""
    return prompt


# ---------------------------------------------------------------------------
# Watsonx.ai client (lazy-initialised)
# ---------------------------------------------------------------------------

_watsonx_model = None   # ibm_watsonx_ai.ModelInference instance


def _get_model():
    """Lazy-load the IBM Granite model via watsonx.ai SDK."""
    global _watsonx_model
    if _watsonx_model is not None:
        return _watsonx_model

    # Validate credentials early and raise a clear error
    if not config.WATSONX_API_KEY:
        raise EnvironmentError(
            "WATSONX_API_KEY is not set. "
            "Add it to your .env file: WATSONX_API_KEY=your_api_key_here"
        )
    if not config.WATSONX_PROJECT_ID:
        raise EnvironmentError(
            "WATSONX_PROJECT_ID is not set. "
            "Add it to your .env file: WATSONX_PROJECT_ID=your_project_id_here"
        )

    from ibm_watsonx_ai import APIClient, Credentials
    from ibm_watsonx_ai.foundation_models import ModelInference

    credentials = Credentials(
        url=config.WATSONX_URL,
        api_key=config.WATSONX_API_KEY,
    )
    client = APIClient(credentials)

    _watsonx_model = ModelInference(
        model_id=config.GRANITE_MODEL_ID,
        api_client=client,
        project_id=config.WATSONX_PROJECT_ID,
    )
    return _watsonx_model


# ---------------------------------------------------------------------------
# Fallback response (when KB has no relevant info)
# ---------------------------------------------------------------------------

NOT_FOUND_RESPONSE = (
    "I'm sorry, I could not find this information in the admission knowledge base. "
    "Please contact the admissions office directly at admissions@democollege.edu "
    "or call the helpline at 1800-XXX-XXXX (demo number) for assistance."
)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def ask(question: str) -> AgentResponse:
    """
    Main entry point for the admission agent.

    Parameters
    ----------
    question : str
        The student's question in natural language.

    Returns
    -------
    AgentResponse
        Contains the answer, source references, and metadata.
    """
    if not question or not question.strip():
        return AgentResponse(
            answer="Please enter a question so I can help you.",
            is_fallback=True,
        )

    # ------------------------------------------------------------------
    # Step 1: Retrieve relevant chunks from the knowledge base
    # ------------------------------------------------------------------
    retrieved = retrieve(question.strip(), top_k=config.TOP_K_CHUNKS)

    # Build source references for display
    sources: List[RetrievedContext] = [
        RetrievedContext(
            source=chunk.source,
            excerpt=chunk.text[:300] + ("…" if len(chunk.text) > 300 else ""),
            score=score,
        )
        for chunk, score in retrieved
    ]

    # ------------------------------------------------------------------
    # Step 2: Fallback — no relevant chunks found
    # ------------------------------------------------------------------
    if not retrieved:
        return AgentResponse(
            answer=NOT_FOUND_RESPONSE,
            sources=[],
            is_fallback=True,
        )

    # ------------------------------------------------------------------
    # Step 3: Build grounded prompt and call IBM Granite
    # ------------------------------------------------------------------
    prompt = _build_prompt(question.strip(), retrieved)

    try:
        model = _get_model()
        # granite-4 uses the chat API; older models fall back gracefully
        messages = [{"role": "user", "content": prompt}]
        response = model.chat(messages=messages)
        # Extract text from chat response structure
        answer = (
            response["choices"][0]["message"]["content"].strip()
            if isinstance(response, dict)
            else str(response).strip()
        )
    except EnvironmentError as exc:
        # Missing credentials — surface a clear, actionable message
        return AgentResponse(
            answer=(
                "⚠️ The AI model is not configured yet.\n\n"
                "To enable AI-generated answers, add your IBM watsonx.ai credentials "
                "to the `.env` file:\n"
                "```\nWATSONX_API_KEY=your_api_key_here\n"
                "WATSONX_PROJECT_ID=your_project_id_here\n```\n\n"
                "You can still browse the retrieved context below."
            ),
            sources=sources,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001
        return AgentResponse(
            answer=(
                "I encountered an error while generating the answer. "
                "Please try again or contact support.\n\n"
                f"Error details: {exc}"
            ),
            sources=sources,
            error=str(exc),
        )

    # ------------------------------------------------------------------
    # Step 4: Return the grounded answer with sources
    # ------------------------------------------------------------------
    return AgentResponse(answer=answer, sources=sources)
