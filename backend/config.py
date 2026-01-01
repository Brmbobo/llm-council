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
