"""
Context Manager
===============
Manages document context for LLM prompt injection with token budgeting.

Token Counting Strategy:
- Uses tiktoken (cl100k_base) for accurate OpenAI-compatible counting
- For other models, this is a reasonable approximation
- Budget is conservative to ensure compatibility across all council models
"""

from dataclasses import dataclass
from typing import List, Optional

import tiktoken

from ..config import MAX_CONTEXT_TOKENS, CONTEXT_INJECTION_TEMPLATE


@dataclass
class DocumentContext:
    """A document prepared for context injection."""
    document_id: str
    filename: str
    text: str
    token_count: int


@dataclass
class InjectionResult:
    """Result of context preparation for injection."""
    formatted_context: str
    total_tokens: int
    documents_included: int
    documents_truncated: int
    overflow_warning: Optional[str] = None


class ContextManager:
    """
    Manages document context with token-aware budgeting.

    Usage:
        manager = ContextManager()
        result = manager.prepare_context(documents, max_tokens=4000)
        full_prompt = result.formatted_context + user_query
    """

    def __init__(self, encoding_name: str = "cl100k_base"):
        """
        Initialize with tiktoken encoding.

        Args:
            encoding_name: tiktoken encoding (cl100k_base for GPT-4/3.5)
        """
        self._encoder = tiktoken.get_encoding(encoding_name)

    def count_tokens(self, text: str) -> int:
        """Count tokens in text using tiktoken."""
        return len(self._encoder.encode(text))

    def prepare_context(
        self,
        documents: List[DocumentContext],
        max_tokens: int = MAX_CONTEXT_TOKENS,
    ) -> InjectionResult:
        """
        Prepare documents for prompt injection within token budget.

        Strategy:
        1. Sort documents by token count (smallest first for max coverage)
        2. Greedily add documents until budget exhausted
        3. Truncate last document if partial fit
        4. Format with template

        Args:
            documents: List of documents to include
            max_tokens: Maximum tokens for context

        Returns:
            InjectionResult with formatted context
        """
        if not documents:
            return InjectionResult(
                formatted_context="",
                total_tokens=0,
                documents_included=0,
                documents_truncated=0
            )

        # Reserve tokens for template overhead
        template_overhead = self.count_tokens(
            CONTEXT_INJECTION_TEMPLATE.format(document_content="")
        )
        available_tokens = max_tokens - template_overhead

        # Sort by size (include more small docs)
        sorted_docs = sorted(documents, key=lambda d: d.token_count)

        included_texts = []
        total_tokens = 0
        documents_included = 0
        documents_truncated = 0

        for doc in sorted_docs:
            if total_tokens >= available_tokens:
                break

            remaining = available_tokens - total_tokens

            if doc.token_count <= remaining:
                # Full document fits
                included_texts.append(
                    f"### {doc.filename}\n{doc.text}"
                )
                total_tokens += doc.token_count
                documents_included += 1
            else:
                # Truncate to fit
                truncated = self._truncate_to_tokens(doc.text, remaining - 20)
                if truncated and len(truncated) > 50:
                    included_texts.append(
                        f"### {doc.filename} (truncated)\n{truncated}"
                    )
                    total_tokens += self.count_tokens(truncated)
                    documents_included += 1
                    documents_truncated += 1
                break

        # Format with template
        document_content = "\n\n---\n\n".join(included_texts)
        formatted = CONTEXT_INJECTION_TEMPLATE.format(
            document_content=document_content
        )

        # Warning if we couldn't include all
        overflow_warning = None
        if documents_included < len(documents):
            overflow_warning = (
                f"Only {documents_included}/{len(documents)} documents included "
                f"due to {max_tokens} token limit"
            )

        return InjectionResult(
            formatted_context=formatted,
            total_tokens=total_tokens + template_overhead,
            documents_included=documents_included,
            documents_truncated=documents_truncated,
            overflow_warning=overflow_warning
        )

    def _truncate_to_tokens(self, text: str, max_tokens: int) -> str:
        """Truncate text to fit within token limit."""
        if max_tokens <= 0:
            return ""

        tokens = self._encoder.encode(text)
        if len(tokens) <= max_tokens:
            return text

        # Reserve space for truncation notice
        truncated_tokens = tokens[:max_tokens - 10]
        truncated = self._encoder.decode(truncated_tokens)

        return truncated + "\n\n[... truncated ...]"
