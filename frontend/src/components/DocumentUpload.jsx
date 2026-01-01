import { useState, useCallback } from 'react';
import './DocumentUpload.css';

const ALLOWED_TYPES = ['.txt', '.pdf', '.md'];
const MAX_SIZE_MB = 10;

export function DocumentUpload({ conversationId, onUploadComplete, disabled }) {
  const [isDragging, setIsDragging] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [error, setError] = useState(null);

  const validateFile = (file) => {
    const ext = '.' + file.name.split('.').pop().toLowerCase();
    if (!ALLOWED_TYPES.includes(ext)) {
      return `File type ${ext} not allowed. Use: ${ALLOWED_TYPES.join(', ')}`;
    }
    if (file.size > MAX_SIZE_MB * 1024 * 1024) {
      return `File too large. Maximum: ${MAX_SIZE_MB}MB`;
    }
    return null;
  };

  const uploadFile = async (file) => {
    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      return;
    }

    setIsUploading(true);
    setError(null);

    try {
      const formData = new FormData();
      formData.append('file', file);

      const response = await fetch(
        `http://localhost:8001/api/conversations/${conversationId}/documents`,
        {
          method: 'POST',
          body: formData,
        }
      );

      if (!response.ok) {
        const err = await response.json();
        throw new Error(err.detail?.message || 'Upload failed');
      }

      const result = await response.json();

      if (onUploadComplete) {
        onUploadComplete(result);
      }

      setError(null);
    } catch (err) {
      setError(err.message);
    } finally {
      setIsUploading(false);
    }
  };

  const handleDragOver = useCallback((e) => {
    e.preventDefault();
    if (!disabled && !isUploading) {
      setIsDragging(true);
    }
  }, [disabled, isUploading]);

  const handleDragLeave = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);
  }, []);

  const handleDrop = useCallback((e) => {
    e.preventDefault();
    setIsDragging(false);

    if (disabled || isUploading) return;

    const files = e.dataTransfer.files;
    if (files.length > 0) {
      uploadFile(files[0]);
    }
  }, [conversationId, disabled, isUploading]);

  const handleClick = () => {
    if (disabled || isUploading) return;

    const input = document.createElement('input');
    input.type = 'file';
    input.accept = ALLOWED_TYPES.join(',');
    input.onchange = (e) => {
      if (e.target.files.length > 0) {
        uploadFile(e.target.files[0]);
      }
    };
    input.click();
  };

  if (!conversationId) {
    return null;
  }

  return (
    <div className="document-upload">
      <div
        className={`dropzone ${isDragging ? 'dragging' : ''} ${isUploading ? 'uploading' : ''} ${disabled ? 'disabled' : ''}`}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
        onClick={handleClick}
      >
        {isUploading ? (
          <div className="upload-progress">
            <div className="spinner"></div>
            <span>Uploading...</span>
          </div>
        ) : (
          <>
            <div className="dropzone-icon">+</div>
            <div className="dropzone-text">
              Drop file or click
            </div>
            <div className="dropzone-hint">
              PDF, TXT, MD (max {MAX_SIZE_MB}MB)
            </div>
          </>
        )}
      </div>

      {error && (
        <div className="upload-error">
          {error}
        </div>
      )}
    </div>
  );
}
