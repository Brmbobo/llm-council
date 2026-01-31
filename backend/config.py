"""Configuration for the LLM Council."""

import os
from pathlib import Path
from typing import Set

from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "mistralai/devstral-2512:free",       # FREE - 256K context, coding expert
    "xiaomi/mimo-v2-flash:free",           # FREE - 256K context, fastest (150 TPS)
    "google/gemini-3-flash-preview",       # 1M context, $0.50/$3
    "minimax/minimax-m2",                  # 128K context, $0.30/$1.20
]

# Chairman model - synthesizes final response (2M context window!)
CHAIRMAN_MODEL = "x-ai/grok-4-1-fast-reasoning"

# OpenRouter API endpoint
OPENROUTER_API_URL = "https://openrouter.ai/api/v1/chat/completions"

# Data directory for conversation storage
DATA_DIR = "data/conversations"

# ═══════════════════════════════════════════════════════════════════════════════
# DOCUMENT UPLOAD CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Storage
DOCUMENTS_DIR = Path("data/documents")
DOCUMENTS_DIR.mkdir(parents=True, exist_ok=True)

# File Constraints
MAX_FILE_SIZE_BYTES: int = 10 * 1024 * 1024  # 10 MB
MAX_FILES_PER_CONVERSATION: int = 10

# Allowed Types (extension -> MIME types)
ALLOWED_FILE_TYPES: dict[str, Set[str]] = {
    ".txt": {"text/plain"},
    ".md": {"text/plain", "text/markdown", "text/x-markdown"},
    ".pdf": {"application/pdf"},
}

# Magic Bytes Signatures
MAGIC_SIGNATURES: dict[str, bytes] = {
    "application/pdf": b"%PDF",
}

# ═══════════════════════════════════════════════════════════════════════════════
# CONTEXT INJECTION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Token Budget (100K for large document uploads)
MAX_CONTEXT_TOKENS: int = 100_000

# Context Template
CONTEXT_INJECTION_TEMPLATE: str = """
<user_documents>
The user has provided the following reference documents. Use this information to inform your response.

{document_content}
</user_documents>

"""

# ═══════════════════════════════════════════════════════════════════════════════
# AGENT PAIR ORCHESTRATION CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

# Agent pairs for parallel execution (Creator, Critic)
# Each pair runs independently and concurrently
AGENT_PAIRS: list[dict[str, str]] = [
    {
        "name": "Pair Alpha",
        "creator": "google/gemini-3-flash-preview",
        "critic": "mistralai/devstral-2512:free",
    },
    {
        "name": "Pair Beta",
        "creator": "minimax/minimax-m2",
        "critic": "xiaomi/mimo-v2-flash:free",
    },
]

# Tester model for validation (high-context reasoning model)
TESTER_MODEL: str = "x-ai/grok-4-1-fast-reasoning"

# ═══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATION SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Iteration limits
MAX_REFINEMENT_ITERATIONS: int = 5           # Dôkladná stratégia - max iterations per pair
CONVERGENCE_THRESHOLD: float = 0.95          # Stop when critic score >= 95%

# Parallel execution
PARALLEL_PAIRS_ENABLED: bool = True          # Run pairs concurrently

# Auto-fix on validation failure
AUTO_FIX_ON_VALIDATION_FAIL: bool = True     # Tester fail → feedback back to pair
MAX_VALIDATION_RETRIES: int = 2              # Max auto-fix attempts

# Timeout settings (milliseconds)
CREATOR_TIMEOUT_MS: int = 120_000            # 2 minutes for creator
CRITIC_TIMEOUT_MS: int = 90_000              # 1.5 minutes for critic
TESTER_TIMEOUT_MS: int = 120_000             # 2 minutes for tester

# ═══════════════════════════════════════════════════════════════════════════════
# UI TRANSPARENCY SETTINGS
# ═══════════════════════════════════════════════════════════════════════════════

# Full transparency mode - shows all iterations, scores, and raw feedback
SHOW_ALL_ITERATIONS: bool = True
SHOW_RAW_FEEDBACK: bool = True
SHOW_ALL_SCORES: bool = True
SHOW_TIMING_INFO: bool = True
