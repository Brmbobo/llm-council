"""
Text Extraction Pipeline
========================
Extract text from various document formats for LLM context injection.

Supported formats:
- PDF: PyMuPDF (fitz) for fast, high-quality extraction
- TXT: chardet for encoding detection
- MD: Pass-through with normalization
"""

import re
from dataclasses import dataclass, field

import chardet
import fitz  # PyMuPDF


@dataclass
class ExtractionResult:
    """Result of text extraction."""
    success: bool
    text: str | None = None
    char_count: int = 0
    page_count: int = 0
    error_code: str | None = None
    error_message: str | None = None
    warnings: list[str] = field(default_factory=list)


class TextExtractionError(Exception):
    """Raised when text extraction fails."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class TextExtractor:
    """
    Multi-format text extraction with quality normalization.

    Usage:
        extractor = TextExtractor()
        result = await extractor.extract(file_bytes, "application/pdf")
    """

    # Minimum text length to consider extraction successful
    MIN_TEXT_LENGTH = 10

    async def extract(
        self,
        file_content: bytes,
        mime_type: str,
    ) -> ExtractionResult:
        """
        Extract text from file based on MIME type.

        Args:
            file_content: Raw file bytes
            mime_type: Detected MIME type

        Returns:
            ExtractionResult with extracted text or error
        """
        extractors = {
            "application/pdf": self._extract_pdf,
            "text/plain": self._extract_text,
            "text/markdown": self._extract_text,
            "text/x-markdown": self._extract_text,
        }

        extractor = extractors.get(mime_type)
        if not extractor:
            return ExtractionResult(
                success=False,
                error_code="UNSUPPORTED_FORMAT",
                error_message=f"No extractor for MIME type: {mime_type}"
            )

        try:
            result = await extractor(file_content)

            # Post-process: normalize whitespace
            if result.success and result.text:
                result.text = self._normalize_text(result.text)
                result.char_count = len(result.text)

                # Warn if extraction seems poor
                if result.char_count < self.MIN_TEXT_LENGTH:
                    result.warnings.append(
                        "Very little text extracted. File may be scanned/image-based."
                    )

            return result

        except Exception as e:
            return ExtractionResult(
                success=False,
                error_code="EXTRACTION_FAILED",
                error_message=str(e)
            )

    async def _extract_pdf(self, content: bytes) -> ExtractionResult:
        """
        Extract text from PDF using PyMuPDF.

        Handles:
        - Multi-page documents
        - Encrypted PDFs (returns error)
        - Image-only PDFs (returns warning)
        """
        try:
            doc = fitz.open(stream=content, filetype="pdf")
        except Exception as e:
            error_str = str(e).lower()
            if "encrypted" in error_str or "password" in error_str:
                return ExtractionResult(
                    success=False,
                    error_code="PDF_ENCRYPTED",
                    error_message="Cannot extract text from password-protected PDF"
                )
            return ExtractionResult(
                success=False,
                error_code="PDF_OPEN_FAILED",
                error_message=f"Failed to open PDF: {str(e)}"
            )

        text_parts = []
        page_count = len(doc)
        warnings = []

        for page_num, page in enumerate(doc):
            text = page.get_text("text")
            if text.strip():
                text_parts.append(f"[Page {page_num + 1}]\n{text}")

        doc.close()

        full_text = "\n\n".join(text_parts)

        # Check if PDF might be scanned (very little text for page count)
        if page_count > 0 and len(full_text) < page_count * 100:
            warnings.append(
                "PDF appears to contain mostly images. Text extraction may be incomplete."
            )

        return ExtractionResult(
            success=True,
            text=full_text,
            page_count=page_count,
            char_count=len(full_text),
            warnings=warnings
        )

    async def _extract_text(self, content: bytes) -> ExtractionResult:
        """
        Extract text from TXT/MD with encoding detection.

        Uses chardet for robust encoding detection.
        """
        # Detect encoding
        detection = chardet.detect(content)
        encoding = detection.get("encoding") or "utf-8"
        confidence = detection.get("confidence") or 0

        warnings = []
        if confidence < 0.8:
            warnings.append(
                f"Low encoding confidence ({confidence:.0%}), assuming {encoding}"
            )

        try:
            text = content.decode(encoding)
        except (UnicodeDecodeError, LookupError):
            # Fallback to utf-8 with error replacement
            text = content.decode("utf-8", errors="replace")
            warnings.append(
                "Encoding issues detected, some characters may be corrupted"
            )

        return ExtractionResult(
            success=True,
            text=text,
            page_count=1,
            char_count=len(text),
            warnings=warnings
        )

    def _normalize_text(self, text: str) -> str:
        """
        Normalize extracted text for LLM consumption.

        - Collapse multiple newlines
        - Remove excessive whitespace
        - Strip leading/trailing whitespace
        """
        # Collapse 3+ newlines to 2
        text = re.sub(r'\n{3,}', '\n\n', text)

        # Remove trailing whitespace per line
        text = '\n'.join(line.rstrip() for line in text.split('\n'))

        # Remove leading/trailing whitespace
        text = text.strip()

        return text
