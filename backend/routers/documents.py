"""
Documents API Router
====================
Handles document upload, listing, and deletion for conversations.
"""

import uuid
from dataclasses import asdict

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from ..services.context_manager import ContextManager
from ..services.document_storage import DocumentStorage, DocumentStorageError
from ..services.file_validator import FileValidator
from ..services.text_extractor import TextExtractor

router = APIRouter(
    prefix="/api/conversations/{conversation_id}/documents",
    tags=["documents"]
)

# Service instances
validator = FileValidator()
extractor = TextExtractor()
storage = DocumentStorage()
context_manager = ContextManager()


class DocumentResponse(BaseModel):
    """Response model for document operations."""
    id: str
    filename: str
    size_bytes: int
    token_count: int
    page_count: int
    warnings: list[str]


class DocumentListResponse(BaseModel):
    """Response model for document listing."""
    documents: list[dict]
    total_tokens: int


class DeleteResponse(BaseModel):
    """Response model for delete operation."""
    deleted: bool


@router.post("", response_model=DocumentResponse)
async def upload_document(
    conversation_id: str,
    file: UploadFile = File(...)
):
    """
    Upload and process a document.

    The document will be validated, text will be extracted, and it will be
    stored for use as context in future queries.
    """
    # Read file content
    content = await file.read()

    # Validate file
    validation = await validator.validate(
        content,
        file.filename or "unnamed",
        file.content_type
    )

    if not validation.is_valid:
        raise HTTPException(
            status_code=400,
            detail={
                "code": validation.error_code,
                "message": validation.error_message
            }
        )

    # Extract text
    extraction = await extractor.extract(content, validation.detected_mime)

    if not extraction.success:
        raise HTTPException(
            status_code=422,
            detail={
                "code": extraction.error_code,
                "message": extraction.error_message
            }
        )

    # Count tokens
    token_count = context_manager.count_tokens(extraction.text or "")

    # Generate document ID
    document_id = uuid.uuid4().hex[:12]

    # Store document
    try:
        metadata = await storage.save_document(
            conversation_id=conversation_id,
            document_id=document_id,
            content=content,
            filename=validation.sanitized_filename,
            mime_type=validation.detected_mime,
            extracted_text=extraction.text or "",
            token_count=token_count,
            page_count=extraction.page_count,
            warnings=extraction.warnings
        )
    except DocumentStorageError as e:
        raise HTTPException(
            status_code=400,
            detail={
                "code": e.code,
                "message": e.message
            }
        ) from e

    return DocumentResponse(
        id=metadata.id,
        filename=metadata.original_filename,
        size_bytes=metadata.size_bytes,
        token_count=metadata.token_count,
        page_count=metadata.page_count,
        warnings=metadata.warnings
    )


@router.get("", response_model=DocumentListResponse)
async def list_documents(conversation_id: str):
    """List all documents for a conversation."""
    docs = await storage.list_documents(conversation_id)

    total_tokens = sum(d.token_count for d in docs)

    return DocumentListResponse(
        documents=[asdict(d) for d in docs],
        total_tokens=total_tokens
    )


@router.delete("/{document_id}", response_model=DeleteResponse)
async def delete_document(conversation_id: str, document_id: str):
    """Delete a document from a conversation."""
    deleted = await storage.delete_document(conversation_id, document_id)

    if not deleted:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DOCUMENT_NOT_FOUND",
                "message": "Document not found"
            }
        )

    return DeleteResponse(deleted=True)


@router.get("/{document_id}/content")
async def get_document_content(conversation_id: str, document_id: str):
    """Get the extracted text content of a document."""
    text = await storage.get_extracted_text(conversation_id, document_id)

    if text is None:
        raise HTTPException(
            status_code=404,
            detail={
                "code": "DOCUMENT_NOT_FOUND",
                "message": "Document not found"
            }
        )

    return {"content": text}
