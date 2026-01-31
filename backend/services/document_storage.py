"""
Document Storage Service
========================
Manages document persistence with conversation-scoped isolation.

Storage Structure:
    data/documents/
    └── {conversation_id}/
        ├── metadata.json          # Document registry
        └── files/
            ├── abc123_report.pdf  # Original files
            └── abc123_report.txt  # Extracted text cache
"""

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import aiofiles

from ..config import DOCUMENTS_DIR, MAX_FILES_PER_CONVERSATION


@dataclass
class DocumentMetadata:
    """Metadata for a stored document."""
    id: str
    original_filename: str
    stored_filename: str
    mime_type: str
    size_bytes: int
    uploaded_at: str
    extraction_status: str
    char_count: int
    token_count: int
    page_count: int
    warnings: list[str]


class DocumentStorageError(Exception):
    """Raised when document storage operation fails."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(message)


class DocumentStorage:
    """
    Async document storage with conversation isolation.

    Usage:
        storage = DocumentStorage()
        metadata = await storage.save_document(...)
        docs = await storage.list_documents(conversation_id)
    """

    def __init__(self):
        self.base_dir = DOCUMENTS_DIR

    def _conversation_dir(self, conversation_id: str) -> Path:
        """Get conversation-specific directory."""
        return self.base_dir / conversation_id

    def _files_dir(self, conversation_id: str) -> Path:
        """Get files subdirectory."""
        return self._conversation_dir(conversation_id) / "files"

    def _metadata_path(self, conversation_id: str) -> Path:
        """Get metadata file path."""
        return self._conversation_dir(conversation_id) / "metadata.json"

    async def save_document(
        self,
        conversation_id: str,
        document_id: str,
        content: bytes,
        filename: str,
        mime_type: str,
        extracted_text: str,
        token_count: int,
        page_count: int,
        warnings: list[str],
    ) -> DocumentMetadata:
        """
        Save document and its extracted text.

        Args:
            conversation_id: Conversation to associate with
            document_id: Unique document ID
            content: Raw file bytes
            filename: Sanitized filename
            mime_type: Detected MIME type
            extracted_text: Extracted text content
            token_count: Token count of extracted text
            page_count: Number of pages (for PDFs)
            warnings: Any extraction warnings

        Returns:
            DocumentMetadata for the saved document
        """
        # Check document limit
        existing = await self.list_documents(conversation_id)
        if len(existing) >= MAX_FILES_PER_CONVERSATION:
            raise DocumentStorageError(
                "MAX_DOCUMENTS_REACHED",
                f"Maximum of {MAX_FILES_PER_CONVERSATION} documents per conversation"
            )

        # Ensure directories exist
        files_dir = self._files_dir(conversation_id)
        files_dir.mkdir(parents=True, exist_ok=True)

        # Save original file
        stored_filename = f"{document_id}_{filename}"
        file_path = files_dir / stored_filename
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(content)

        # Save extracted text
        stem = Path(filename).stem
        text_filename = f"{document_id}_{stem}.txt"
        text_path = files_dir / text_filename
        async with aiofiles.open(text_path, 'w', encoding='utf-8') as f:
            await f.write(extracted_text)

        # Create metadata
        metadata = DocumentMetadata(
            id=document_id,
            original_filename=filename,
            stored_filename=stored_filename,
            mime_type=mime_type,
            size_bytes=len(content),
            uploaded_at=datetime.utcnow().isoformat() + "Z",
            extraction_status="success",
            char_count=len(extracted_text),
            token_count=token_count,
            page_count=page_count,
            warnings=warnings
        )

        # Update metadata registry
        await self._add_to_registry(conversation_id, metadata)

        return metadata

    async def list_documents(
        self,
        conversation_id: str
    ) -> list[DocumentMetadata]:
        """List all documents for conversation."""
        metadata_path = self._metadata_path(conversation_id)
        if not metadata_path.exists():
            return []

        async with aiofiles.open(metadata_path, encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        return [DocumentMetadata(**doc) for doc in data.get("documents", [])]

    async def get_document(
        self,
        conversation_id: str,
        document_id: str
    ) -> DocumentMetadata | None:
        """Get a specific document's metadata."""
        docs = await self.list_documents(conversation_id)
        return next((d for d in docs if d.id == document_id), None)

    async def get_extracted_text(
        self,
        conversation_id: str,
        document_id: str
    ) -> str | None:
        """Get extracted text for document."""
        doc = await self.get_document(conversation_id, document_id)
        if not doc:
            return None

        stem = Path(doc.original_filename).stem
        text_filename = f"{document_id}_{stem}.txt"
        text_path = self._files_dir(conversation_id) / text_filename

        if not text_path.exists():
            return None

        async with aiofiles.open(text_path, encoding='utf-8') as f:
            return await f.read()

    async def delete_document(
        self,
        conversation_id: str,
        document_id: str
    ) -> bool:
        """Delete document and its files."""
        doc = await self.get_document(conversation_id, document_id)
        if not doc:
            return False

        files_dir = self._files_dir(conversation_id)

        # Delete original file
        original_path = files_dir / doc.stored_filename
        if original_path.exists():
            original_path.unlink()

        # Delete extracted text
        stem = Path(doc.original_filename).stem
        text_filename = f"{document_id}_{stem}.txt"
        text_path = files_dir / text_filename
        if text_path.exists():
            text_path.unlink()

        # Update registry
        await self._remove_from_registry(conversation_id, document_id)

        return True

    async def _add_to_registry(
        self,
        conversation_id: str,
        metadata: DocumentMetadata
    ):
        """Add document to metadata registry."""
        metadata_path = self._metadata_path(conversation_id)

        if metadata_path.exists():
            async with aiofiles.open(metadata_path, encoding='utf-8') as f:
                content = await f.read()
                data = json.loads(content)
        else:
            data = {
                "documents": [],
                "total_size_bytes": 0,
                "created_at": datetime.utcnow().isoformat() + "Z"
            }

        data["documents"].append(asdict(metadata))
        data["total_size_bytes"] += metadata.size_bytes

        async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
            await f.write(json.dumps(data, indent=2))

    async def _remove_from_registry(
        self,
        conversation_id: str,
        document_id: str
    ):
        """Remove document from registry."""
        metadata_path = self._metadata_path(conversation_id)
        if not metadata_path.exists():
            return

        async with aiofiles.open(metadata_path, encoding='utf-8') as f:
            content = await f.read()
            data = json.loads(content)

        # Find and remove document
        original_docs = data.get("documents", [])
        removed_doc = next((d for d in original_docs if d["id"] == document_id), None)

        if removed_doc:
            data["documents"] = [d for d in original_docs if d["id"] != document_id]
            data["total_size_bytes"] -= removed_doc.get("size_bytes", 0)

            async with aiofiles.open(metadata_path, 'w', encoding='utf-8') as f:
                await f.write(json.dumps(data, indent=2))
