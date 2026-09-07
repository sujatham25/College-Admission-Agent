"""
config.py
---------
Centralised configuration for the College Admission Agent.
Credentials are loaded from a .env file — never hard-coded.

⚠️  DEMO PROJECT — IBM AICTE Internship 2026
"""

import os
from dotenv import load_dotenv

# Load variables from .env (if present) into the environment
load_dotenv()

# ---------------------------------------------------------------------------
# IBM watsonx.ai credentials
# ---------------------------------------------------------------------------

WATSONX_API_KEY: str = os.getenv("WATSONX_API_KEY", "")
WATSONX_PROJECT_ID: str = os.getenv("WATSONX_PROJECT_ID", "")
WATSONX_URL: str = os.getenv(
    "WATSONX_URL", "https://us-south.ml.cloud.ibm.com"
)

# ---------------------------------------------------------------------------
# IBM Granite model
# ---------------------------------------------------------------------------

# IBM Granite 4 H Small — as specified in project credentials
GRANITE_MODEL_ID: str = "ibm/granite-4-h-small"

# Generation parameters
GRANITE_MAX_NEW_TOKENS: int = 512
GRANITE_TEMPERATURE: float = 0.0        # deterministic; reduces hallucinations
GRANITE_TOP_P: float = 1.0
GRANITE_REPETITION_PENALTY: float = 1.1

# ---------------------------------------------------------------------------
# RAG settings
# ---------------------------------------------------------------------------

TOP_K_CHUNKS: int = 6         # number of retrieved chunks to include in the prompt
MIN_CONTEXT_SCORE: float = 0.05  # minimum similarity score to consider a chunk relevant
