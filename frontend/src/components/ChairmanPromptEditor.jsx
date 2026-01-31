import { useState } from 'react';
import './ChairmanPromptEditor.css';

export default function ChairmanPromptEditor({
  defaultPrompt,
  customPrompt,
  onPromptChange
}) {
  const [isExpanded, setIsExpanded] = useState(false);

  const currentPrompt = customPrompt || defaultPrompt || '';
  const isCustomized = customPrompt !== null && customPrompt !== undefined;

  const handleReset = () => {
    onPromptChange(null);
  };

  return (
    <div className="chairman-editor">
      <button
        className={`chairman-toggle ${isExpanded ? 'expanded' : ''}`}
        onClick={() => setIsExpanded(!isExpanded)}
      >
        <span className="toggle-icon">{isExpanded ? '▼' : '▶'}</span>
        <span className="toggle-label">Chairman Prompt</span>
        {isCustomized && (
          <span className="customized-badge">Customized</span>
        )}
      </button>

      {isExpanded && (
        <div className="chairman-content">
          <div className="chairman-header">
            <span className="chairman-hint">
              The chairman synthesizes all responses into a final answer
            </span>
            {isCustomized && (
              <button className="reset-btn" onClick={handleReset}>
                Reset to Default
              </button>
            )}
          </div>
          <textarea
            className="chairman-textarea"
            value={currentPrompt}
            onChange={(e) => onPromptChange(e.target.value || null)}
            placeholder="Enter custom chairman prompt..."
            rows={8}
          />
        </div>
      )}
    </div>
  );
}
