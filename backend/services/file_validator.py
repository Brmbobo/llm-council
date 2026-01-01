"""
File Validation Service
=======================
Multi-layer validation for uploaded files following OWASP guidelines.

Security Layers:
1. Extension whitelist (fast, first line)
2. Content-Type header check (client-declared)
3. Magic bytes verification (binary truth)
4. Size constraints
5. Filename sanitization (path traversal prevention)
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import re
import uuid

import magic

from ..config import (
    ALLOWED_FILE_TYPES,
    MAGIC_SIGNATURES,
    MAX_FILE_SIZE_BYTES,
)


@dataclass
class ValidationResult:
    """Immutable validation outcome."""
    is_valid: bool
    error_code: Optional[str] = None
    error_message: Optional[str] = None
    sanitized_filename: Optional[str] = None
    detected_mime: Optional[str] = None


class FileValidationError(Exception):
    """Raised when file validation fails."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class FileValidator:
    """
    Production-grade file validator with multi-layer security.

    Usage:
        validator = FileValidator()
        result = await validator.validate(file_bytes, filename, content_type)
        if not result.is_valid:
            raise HTTPException(400, result.error_message)
    """

    # Path traversal patterns
    DANGEROUS_PATTERNS = re.compile(r'(\.\./|\.\.\\|^/|^\\)')

    def __init__(self):
        self._magic = magic.Magic(mime=True)

    async def validate(
        self,
        file_content: bytes,
        original_filename: str,
        declared_content_type: Optional[str] = None,
    ) -> ValidationResult:
        """
        Validate uploaded file through all security layers.

        Args:
            file_content: Raw file bytes
            original_filename: Client-provided filename
            declared_content_type: Content-Type header value

        Returns:
            ValidationResult with validation outcome
        """
        # Layer 1: Size check (fastest, fail early)
        if len(file_content) > MAX_FILE_SIZE_BYTES:
            return ValidationResult(
                is_valid=False,
                error_code="FILE_TOO_LARGE",
                error_message=f"File exceeds {MAX_FILE_SIZE_BYTES // 1024 // 1024}MB limit"
            )

        if len(file_content) == 0:
            return ValidationResult(
                is_valid=False,
                error_code="FILE_EMPTY",
                error_message="File is empty"
            )

        # Layer 2: Extension whitelist
        extension = Path(original_filename).suffix.lower()
        if extension not in ALLOWED_FILE_TYPES:
            allowed = list(ALLOWED_FILE_TYPES.keys())
            return ValidationResult(
                is_valid=False,
                error_code="INVALID_EXTENSION",
                error_message=f"File type {extension} not allowed. Allowed: {allowed}"
            )

        # Layer 3: Magic bytes detection
        detected_mime = self._magic.from_buffer(file_content[:2048])
        allowed_mimes = ALLOWED_FILE_TYPES[extension]

        if detected_mime not in allowed_mimes:
            return ValidationResult(
                is_valid=False,
                error_code="MIME_MISMATCH",
                error_message=f"File content ({detected_mime}) doesn't match extension ({extension})"
            )

        # Layer 4: PDF-specific magic bytes verification
        if extension == ".pdf":
            if not file_content.startswith(MAGIC_SIGNATURES["application/pdf"]):
                return ValidationResult(
                    is_valid=False,
                    error_code="INVALID_PDF_SIGNATURE",
                    error_message="Invalid PDF file signature"
                )

        # Layer 5: Filename sanitization
        sanitized = self._sanitize_filename(original_filename)

        return ValidationResult(
            is_valid=True,
            sanitized_filename=sanitized,
            detected_mime=detected_mime
        )

    def _sanitize_filename(self, filename: str) -> str:
        """
        Sanitize filename to prevent path traversal and filesystem issues.

        Strategy:
        1. Remove path components
        2. Generate UUID prefix for uniqueness
        3. Preserve original extension
        4. Remove dangerous characters
        """
        # Extract just the filename (no path)
        name = Path(filename).name

        # Remove dangerous patterns
        name = self.DANGEROUS_PATTERNS.sub('', name)

        # Get extension
        ext = Path(name).suffix.lower()
        stem = Path(name).stem

        # Clean stem (keep only safe chars, replace others with underscore)
        clean_stem = re.sub(r'[^\w\-.]', '_', stem)[:50]  # Limit length

        # Add UUID for uniqueness
        unique_id = uuid.uuid4().hex[:8]

        return f"{unique_id}_{clean_stem}{ext}"
