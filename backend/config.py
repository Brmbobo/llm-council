"""Configuration for the LLM Council."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# OpenRouter API key
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

# Council members - list of OpenRouter model identifiers
COUNCIL_MODELS = [
    "openai/gpt-4.1-nano",       # FREE - 256K context, coding expert
    "xiaomi/mimo-v2-flash:free",           # FREE - 256K context, fastest (150 TPS)
    "x-ai/grok-4.1-fast",           # FREE - 256K context, fastest (150 TPS)
    "meta-llama/llama-4-maverick",       # 1M context, $0.50/$3
    "minimax/minimax-m2",                  # 128K context, $0.30/$1.20
]

# Chairman model - synthesizes final response (2M context window!)
CHAIRMAN_MODEL = "google/gemini-3-flash-preview"

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
ALLOWED_FILE_TYPES: dict[str, set[str]] = {
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
# COUNCIL ROLES CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════

COUNCIL_ROLES = {
    "technical": {
        "name": "Technical Critic",
        "icon": "code",
        "default_prompt": """You are a TECHNICAL CRITIC. Evaluate responses from a technical perspective, focusing on:
- Technical accuracy and correctness
- Code quality and best practices (if applicable)
- Performance implications and efficiency
- Security considerations and vulnerabilities
- Scalability and maintainability
- Adherence to industry standards"""
    },
    "marketing": {
        "name": "Marketing Critic",
        "icon": "megaphone",
        "default_prompt": """You are a MARKETING CRITIC. Evaluate responses from a marketing perspective, focusing on:
- Market positioning and competitive differentiation
- Target audience alignment and messaging clarity
- Brand consistency and voice
- Persuasion techniques and call-to-action effectiveness
- Value proposition clarity
- Customer journey optimization"""
    },
    "business": {
        "name": "Business Critic",
        "icon": "briefcase",
        "default_prompt": """You are a BUSINESS CRITIC. Evaluate responses from a business perspective, focusing on:
- ROI and cost-benefit analysis
- Feasibility and resource requirements
- Risk assessment and mitigation strategies
- Scalability and growth potential
- Competitive landscape considerations
- Strategic alignment with business objectives"""
    },
    "ux": {
        "name": "UX/Design Critic",
        "icon": "palette",
        "default_prompt": """You are a UX/DESIGN CRITIC. Evaluate responses from a user experience perspective, focusing on:
- User experience and accessibility (WCAG compliance)
- Intuitiveness and learning curve
- Visual hierarchy and information architecture
- User journey and flow optimization
- Emotional design and user satisfaction
- Mobile-first and responsive considerations"""
    },
    "pragmatism": {
        "name": "Pragmatism Critic",
        "icon": "tools",
        "default_prompt": """You are a PRAGMATISM CRITIC. Evaluate responses from a practical implementation perspective, focusing on:
- Implementation feasibility with available resources
- Time and budget constraints
- Quick wins vs long-term value trade-offs
- Maintenance burden and technical debt
- Team skill requirements
- Realistic timelines and milestones"""
    }
}

# Default active roles (can be overridden per request)
DEFAULT_ACTIVE_ROLES = ["technical", "business", "pragmatism"]

# Default chairman synthesis prompt
DEFAULT_CHAIRMAN_PROMPT = """You are the Chairman of an LLM Council. Multiple AI models have provided responses to a user's question, and then evaluated each other's responses from their specialized perspectives.

Your task as Chairman is to synthesize all of this information into a single, comprehensive, accurate answer to the user's original question. Consider:
- The individual responses and their unique insights
- The peer evaluations and what they reveal about response quality
- Any patterns of agreement or disagreement across perspectives
- The strengths highlighted and weaknesses identified by each critic role

Provide a clear, well-reasoned final answer that represents the council's collective wisdom:"""

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
MAX_REFINEMENT_ITERATIONS: int = 5           # Max iterations per pair
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
