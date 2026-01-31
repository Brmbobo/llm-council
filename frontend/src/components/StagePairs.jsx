import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './StagePairs.css';

/**
 * StagePairs - Full Transparency View of Agent Pair Results
 *
 * Displays all iterations, scores, and raw feedback for each pair.
 * Shows the Creator-Critic refinement process.
 */
export default function StagePairs({ pairs }) {
  const [activePair, setActivePair] = useState(0);
  const [expandedIterations, setExpandedIterations] = useState({});

  if (!pairs || pairs.length === 0) {
    return null;
  }

  const toggleIteration = (pairIndex, iterIndex) => {
    const key = `${pairIndex}-${iterIndex}`;
    setExpandedIterations(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const getStatusIcon = (status) => {
    switch (status) {
      case 'converged': return '✓';
      case 'max_iterations': return '⚠';
      case 'continue': return '→';
      case 'error': return '✗';
      default: return '?';
    }
  };

  const getStatusClass = (status) => {
    switch (status) {
      case 'converged': return 'status-converged';
      case 'max_iterations': return 'status-max';
      case 'continue': return 'status-continue';
      case 'error': return 'status-error';
      default: return '';
    }
  };

  const formatTime = (ms) => {
    if (ms < 1000) return `${ms}ms`;
    return `${(ms / 1000).toFixed(1)}s`;
  };

  const pair = pairs[activePair];

  return (
    <div className="stage stage-pairs">
      <h3 className="stage-title">Stage 1: Agent Pairs (Creator-Critic)</h3>

      {/* Pair Tabs */}
      <div className="tabs">
        {pairs.map((p, index) => (
          <button
            key={index}
            className={`tab ${activePair === index ? 'active' : ''} ${p.converged ? 'converged' : ''}`}
            onClick={() => setActivePair(index)}
          >
            <span className="tab-name">{p.pair_name}</span>
            <span className={`tab-status ${getStatusClass(p.converged ? 'converged' : 'max_iterations')}`}>
              {p.converged ? '✓' : `${p.total_iterations}it`}
            </span>
          </button>
        ))}
      </div>

      {/* Pair Details */}
      <div className="pair-content">
        {/* Pair Header */}
        <div className="pair-header">
          <div className="pair-models">
            <span className="model-label">Creator:</span>
            <span className="model-name">{pair.creator_model}</span>
            <span className="model-label">Critic:</span>
            <span className="model-name">{pair.critic_model}</span>
          </div>
          <div className="pair-stats">
            <div className="stat">
              <span className="stat-label">Iterations</span>
              <span className="stat-value">{pair.total_iterations}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Final Score</span>
              <span className={`stat-value ${pair.final_score >= 0.95 ? 'score-high' : pair.final_score >= 0.8 ? 'score-medium' : 'score-low'}`}>
                {(pair.final_score * 100).toFixed(0)}%
              </span>
            </div>
            <div className="stat">
              <span className="stat-label">Time</span>
              <span className="stat-value">{formatTime(pair.total_time_ms)}</span>
            </div>
            <div className="stat">
              <span className="stat-label">Status</span>
              <span className={`stat-value ${pair.converged ? 'status-converged' : 'status-max'}`}>
                {pair.converged ? 'Converged' : 'Max Iterations'}
              </span>
            </div>
          </div>
        </div>

        {/* Score Progression */}
        <div className="score-progression">
          <span className="progression-label">Score Progression:</span>
          <div className="progression-bar">
            {pair.iterations.map((iter, idx) => (
              <div
                key={idx}
                className={`progression-point ${iter.status === 'converged' ? 'converged' : ''}`}
                style={{ left: `${(idx / (pair.iterations.length - 1 || 1)) * 100}%` }}
                title={`Iteration ${iter.iteration_number}: ${(iter.critic_score * 100).toFixed(0)}%`}
              >
                <span className="point-score">{(iter.critic_score * 100).toFixed(0)}%</span>
              </div>
            ))}
            <div className="progression-line" />
            <div
              className="threshold-line"
              style={{ left: '95%' }}
              title="95% Convergence Threshold"
            />
          </div>
        </div>

        {/* Iterations List */}
        <div className="iterations-list">
          <h4 className="iterations-title">Iteration History</h4>
          {pair.iterations.map((iter, idx) => {
            const isExpanded = expandedIterations[`${activePair}-${idx}`];
            return (
              <div key={idx} className={`iteration ${getStatusClass(iter.status)}`}>
                <div
                  className="iteration-header"
                  onClick={() => toggleIteration(activePair, idx)}
                >
                  <div className="iteration-info">
                    <span className={`iteration-status ${getStatusClass(iter.status)}`}>
                      {getStatusIcon(iter.status)}
                    </span>
                    <span className="iteration-number">Iteration {iter.iteration_number}</span>
                    <span className={`iteration-score ${iter.critic_score >= 0.95 ? 'score-high' : iter.critic_score >= 0.8 ? 'score-medium' : 'score-low'}`}>
                      {(iter.critic_score * 100).toFixed(0)}%
                    </span>
                  </div>
                  <div className="iteration-meta">
                    <span className="iteration-time">
                      Creator: {formatTime(iter.creator_time_ms)} | Critic: {formatTime(iter.critic_time_ms)}
                    </span>
                    <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>
                      {isExpanded ? '▼' : '▶'}
                    </span>
                  </div>
                </div>

                {isExpanded && (
                  <div className="iteration-details">
                    <div className="detail-section">
                      <h5 className="detail-title">Creator Response</h5>
                      <div className="detail-content markdown-content">
                        <ReactMarkdown>{iter.creator_response}</ReactMarkdown>
                      </div>
                    </div>
                    <div className="detail-section critic-section">
                      <h5 className="detail-title">Critic Feedback</h5>
                      <div className="detail-content markdown-content">
                        <ReactMarkdown>{iter.critic_feedback}</ReactMarkdown>
                      </div>
                    </div>
                  </div>
                )}
              </div>
            );
          })}
        </div>

        {/* Final Response */}
        <div className="final-response">
          <h4 className="final-title">Final Response from {pair.pair_name}</h4>
          <div className="final-content markdown-content">
            <ReactMarkdown>{pair.final_response}</ReactMarkdown>
          </div>
        </div>

        {/* Error Display */}
        {pair.error && (
          <div className="pair-error">
            <span className="error-icon">⚠</span>
            <span className="error-message">{pair.error}</span>
          </div>
        )}
      </div>
    </div>
  );
}
