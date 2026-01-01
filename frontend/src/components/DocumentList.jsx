import { useState, useEffect } from 'react';
import './DocumentList.css';

export function DocumentList({ conversationId, refreshTrigger }) {
  const [documents, setDocuments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [totalTokens, setTotalTokens] = useState(0);

  useEffect(() => {
    if (conversationId) {
      loadDocuments();
    } else {
      setDocuments([]);
      setTotalTokens(0);
    }
  }, [conversationId, refreshTrigger]);

  const loadDocuments = async () => {
    if (!conversationId) return;

    setLoading(true);
    try {
      const response = await fetch(
        `http://localhost:8001/api/conversations/${conversationId}/documents`
      );
      const data = await response.json();
      setDocuments(data.documents || []);
      setTotalTokens(data.total_tokens || 0);
    } catch (err) {
      console.error('Failed to load documents:', err);
    } finally {
      setLoading(false);
    }
  };

  const deleteDocument = async (docId) => {
    try {
      const response = await fetch(
        `http://localhost:8001/api/conversations/${conversationId}/documents/${docId}`,
        { method: 'DELETE' }
      );

      if (response.ok) {
        setDocuments(docs => docs.filter(d => d.id !== docId));
        // Recalculate total tokens
        const deletedDoc = documents.find(d => d.id === docId);
        if (deletedDoc) {
          setTotalTokens(prev => prev - deletedDoc.token_count);
        }
      }
    } catch (err) {
      console.error('Failed to delete:', err);
    }
  };

  const formatFileSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  if (loading) {
    return <div className="documents-loading">Loading documents...</div>;
  }

  if (documents.length === 0) {
    return null;
  }

  return (
    <div className="document-list">
      <div className="document-list-header">
        <span className="document-list-title">
          Attached Documents ({documents.length})
        </span>
        <span className="document-list-tokens">
          {totalTokens.toLocaleString()} tokens
        </span>
      </div>

      <ul className="document-items">
        {documents.map(doc => (
          <li key={doc.id} className="document-item">
            <div className="document-info">
              <span className="document-name" title={doc.original_filename}>
                {doc.original_filename}
              </span>
              <span className="document-meta">
                {doc.page_count > 1 ? `${doc.page_count} pages` : '1 page'}
                {' • '}
                {formatFileSize(doc.size_bytes)}
                {' • '}
                {doc.token_count.toLocaleString()} tokens
              </span>
              {doc.warnings && doc.warnings.length > 0 && (
                <span className="document-warning" title={doc.warnings.join(', ')}>
                  Warning
                </span>
              )}
            </div>
            <button
              className="delete-btn"
              onClick={() => deleteDocument(doc.id)}
              title="Remove document"
            >
              x
            </button>
          </li>
        ))}
      </ul>

      {totalTokens > 3500 && (
        <div className="token-warning">
          High token count may affect response quality
        </div>
      )}
    </div>
  );
}
