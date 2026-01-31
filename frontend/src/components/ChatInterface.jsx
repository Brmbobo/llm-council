import { useState, useEffect, useRef } from 'react';
import ReactMarkdown from 'react-markdown';
import Stage1 from './Stage1';
import Stage2 from './Stage2';
import Stage3 from './Stage3';
import StagePairs from './StagePairs';
import StageValidation from './StageValidation';
import { DocumentUpload } from './DocumentUpload';
import { DocumentList } from './DocumentList';
import './ChatInterface.css';

export default function ChatInterface({
  conversation,
  onSendMessage,
  isLoading,
}) {
  const [input, setInput] = useState('');
  const [documentRefresh, setDocumentRefresh] = useState(0);
  const messagesEndRef = useRef(null);

  const handleDocumentUpload = (result) => {
    // Trigger refresh of document list
    setDocumentRefresh(prev => prev + 1);
  };

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [conversation]);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (input.trim() && !isLoading) {
      onSendMessage(input);
      setInput('');
    }
  };

  const handleKeyDown = (e) => {
    // Submit on Enter (without Shift)
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit(e);
    }
  };

  if (!conversation) {
    return (
      <div className="chat-interface">
        <div className="empty-state">
          <h2>Welcome to LLM Council</h2>
          <p>Create a new conversation to get started</p>
        </div>
      </div>
    );
  }

  return (
    <div className="chat-interface">
      <div className="messages-container">
        {conversation.messages.length === 0 ? (
          <div className="empty-state">
            <h2>Start a conversation</h2>
            <p>Ask a question to consult the LLM Council</p>
          </div>
        ) : (
          conversation.messages.map((msg, index) => (
            <div key={index} className="message-group">
              {msg.role === 'user' ? (
                <div className="user-message">
                  <div className="message-label">You</div>
                  <div className="message-content">
                    <div className="markdown-content">
                      <ReactMarkdown>{msg.content}</ReactMarkdown>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="assistant-message">
                  <div className="message-label">LLM Council</div>

                  {/* Enhanced Mode: Agent Pairs */}
                  {msg.loading?.pairs && (
                    <div className="stage-loading enhanced">
                      <div className="spinner"></div>
                      <span>Running Stage 1: Agent Pairs (Creator-Critic iterations)...</span>
                      {msg.loading.pairs_progress && (
                        <div className="progress-detail">
                          {msg.loading.pairs_progress}
                        </div>
                      )}
                    </div>
                  )}
                  {msg.stage_pairs && <StagePairs pairs={msg.stage_pairs} />}

                  {/* Enhanced Mode: Validation */}
                  {msg.loading?.validation && (
                    <div className="stage-loading enhanced">
                      <div className="spinner"></div>
                      <span>Running Stage 2: Tester Validation + Auto-Fix...</span>
                    </div>
                  )}
                  {msg.stage_validation && <StageValidation validation={msg.stage_validation} />}

                  {/* Traditional Mode: Stage 1 */}
                  {msg.loading?.stage1 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage 1: Collecting individual responses...</span>
                    </div>
                  )}
                  {msg.stage1 && !msg.stage_pairs && <Stage1 responses={msg.stage1} />}

                  {/* Traditional/Enhanced Mode: Stage 2/3 Peer Rankings */}
                  {msg.loading?.stage2 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage {msg.stage_pairs ? '3' : '2'}: Peer rankings...</span>
                    </div>
                  )}
                  {msg.stage2 && !msg.stage_rankings && (
                    <Stage2
                      rankings={msg.stage2}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                    />
                  )}
                  {msg.stage_rankings && (
                    <Stage2
                      rankings={msg.stage_rankings}
                      labelToModel={msg.metadata?.label_to_model}
                      aggregateRankings={msg.metadata?.aggregate_rankings}
                      stageNumber={3}
                    />
                  )}

                  {/* Traditional/Enhanced Mode: Final Synthesis */}
                  {msg.loading?.stage3 && (
                    <div className="stage-loading">
                      <div className="spinner"></div>
                      <span>Running Stage {msg.stage_pairs ? '4' : '3'}: Final synthesis...</span>
                    </div>
                  )}
                  {msg.stage3 && <Stage3 finalResponse={msg.stage3} stageNumber={msg.stage_pairs ? 4 : 3} />}
                  {msg.stage_synthesis && <Stage3 finalResponse={msg.stage_synthesis} stageNumber={4} />}
                </div>
              )}
            </div>
          ))
        )}

        {isLoading && (
          <div className="loading-indicator">
            <div className="spinner"></div>
            <span>Consulting the council...</span>
          </div>
        )}

        <div ref={messagesEndRef} />
      </div>

      {conversation.messages.length === 0 && (
        <div className="input-section">
          <div className="documents-section">
            <DocumentList
              conversationId={conversation.id}
              refreshTrigger={documentRefresh}
            />
            <DocumentUpload
              conversationId={conversation.id}
              onUploadComplete={handleDocumentUpload}
              disabled={isLoading}
            />
          </div>

          <form className="input-form" onSubmit={handleSubmit}>
            <textarea
              className="message-input"
              placeholder="Ask your question... (Shift+Enter for new line, Enter to send)"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              disabled={isLoading}
              rows={3}
            />
            <button
              type="submit"
              className="send-button"
              disabled={!input.trim() || isLoading}
            >
              Send
            </button>
          </form>
        </div>
      )}
    </div>
  );
}
