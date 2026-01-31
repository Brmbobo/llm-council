import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './StageValidation.css';

/**
 * StageValidation - Full Transparency Tester Validation View
 *
 * Displays validation results, scores, issues, strengths,
 * and auto-fix history for each pair.
 */
export default function StageValidation({ validation }) {
  const [activePair, setActivePair] = useState(0);
  const [expandedFixes, setExpandedFixes] = useState({});

  if (!validation || !validation.results || validation.results.length === 0) {
    return null;
  }

  const toggleFix = (pairIndex, fixIndex) => {
    const key = `${pairIndex}-${fixIndex}`;
    setExpandedFixes(prev => ({
      ...prev,
      [key]: !prev[key]
    }));
  };

  const result = validation.results[activePair];

  const ScoreBar = ({ label, value, max = 1 }) => {
    const percentage = (value / max) * 100;
    const getColor = (pct) => {
      if (pct >= 95) return '#10b981';
      if (pct >= 80) return '#f59e0b';
      if (pct >= 60) return '#f97316';
      return '#ef4444';
    };

    return (
      <div className="score-bar">
        <div className="score-bar-label">{label}</div>
        <div className="score-bar-track">
          <div
            className="score-bar-fill"
            style={{
              width: `${percentage}%`,
              backgroundColor: getColor(percentage)
            }}
          />
        </div>
        <div className="score-bar-value">{(value * 100).toFixed(0)}%</div>
      </div>
    );
  };

  return (
    <div className="stage stage-validation">
      <h3 className="stage-title">Stage 2: Tester Validation</h3>

      {/* Summary Banner */}
      <div className={`validation-summary ${validation.all_passed ? 'all-passed' : 'some-failed'}`}>
        <div className="summary-icon">
          {validation.all_passed ? '✓' : '⚠'}
        </div>
        <div className="summary-content">
          <div className="summary-title">
            {validation.all_passed ? 'All Pairs Passed Validation' : 'Validation Results'}
          </div>
          <div className="summary-text">{validation.recommendation}</div>
        </div>
        <div className="summary-best">
          <span className="best-label">Best Pair:</span>
          <span className="best-name">{validation.best_pair}</span>
          <span className={`best-score ${validation.best_score >= 0.95 ? 'score-high' : 'score-medium'}`}>
            {(validation.best_score * 100).toFixed(0)}%
          </span>
        </div>
      </div>

      {/* Pair Tabs */}
      <div className="tabs">
        {validation.results.map((r, index) => (
          <button
            key={index}
            className={`tab ${activePair === index ? 'active' : ''} ${r.passed ? 'passed' : 'failed'}`}
            onClick={() => setActivePair(index)}
          >
            <span className="tab-name">{r.pair_name}</span>
            <span className={`tab-status ${r.passed ? 'status-pass' : 'status-fail'}`}>
              {r.passed ? '✓ PASS' : '✗ FAIL'}
            </span>
          </button>
        ))}
      </div>

      {/* Validation Details */}
      <div className="validation-content">
        {/* Score Cards */}
        <div className="scores-section">
          <h4 className="section-title">Test Scores</h4>

          <div className="scores-grid">
            <div className="score-card initial">
              <div className="score-card-header">Initial Test</div>
              <ScoreBar label="Accuracy" value={result.initial_scores.accuracy} />
              <ScoreBar label="Logic" value={result.initial_scores.logic} />
              <ScoreBar label="Completeness" value={result.initial_scores.completeness} />
              <ScoreBar label="Clarity" value={result.initial_scores.clarity} />
              <div className="score-card-total">
                <span>Overall:</span>
                <span className={`total-value ${result.initial_scores.overall >= 0.95 ? 'score-high' : result.initial_scores.overall >= 0.8 ? 'score-medium' : 'score-low'}`}>
                  {(result.initial_scores.overall * 100).toFixed(0)}%
                </span>
              </div>
            </div>

            {result.auto_fix_attempts.length > 0 && (
              <div className="score-card final">
                <div className="score-card-header">After Auto-Fix</div>
                <ScoreBar label="Accuracy" value={result.final_scores.accuracy || result.initial_scores.accuracy} />
                <ScoreBar label="Logic" value={result.final_scores.logic || result.initial_scores.logic} />
                <ScoreBar label="Completeness" value={result.final_scores.completeness || result.initial_scores.completeness} />
                <ScoreBar label="Clarity" value={result.final_scores.clarity || result.initial_scores.clarity} />
                <div className="score-card-total">
                  <span>Overall:</span>
                  <span className={`total-value ${result.final_scores.overall >= 0.95 ? 'score-high' : result.final_scores.overall >= 0.8 ? 'score-medium' : 'score-low'}`}>
                    {(result.final_scores.overall * 100).toFixed(0)}%
                  </span>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Issues & Strengths */}
        <div className="feedback-section">
          <div className="feedback-column issues">
            <h4 className="section-title">Issues Found</h4>
            {result.issues.length > 0 ? (
              <ul className="feedback-list">
                {result.issues.map((issue, idx) => (
                  <li key={idx} className="feedback-item issue-item">
                    <span className="feedback-icon">!</span>
                    {issue}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="no-items">No issues found</p>
            )}
          </div>

          <div className="feedback-column strengths">
            <h4 className="section-title">Strengths</h4>
            {result.strengths.length > 0 ? (
              <ul className="feedback-list">
                {result.strengths.map((strength, idx) => (
                  <li key={idx} className="feedback-item strength-item">
                    <span className="feedback-icon">+</span>
                    {strength}
                  </li>
                ))}
              </ul>
            ) : (
              <p className="no-items">No strengths listed</p>
            )}
          </div>
        </div>

        {/* Raw Feedback */}
        <div className="raw-feedback-section">
          <h4 className="section-title">Tester's Detailed Feedback</h4>
          <div className="raw-feedback markdown-content">
            <ReactMarkdown>{result.raw_feedback}</ReactMarkdown>
          </div>
        </div>

        {/* Auto-Fix History */}
        {result.auto_fix_attempts.length > 0 && (
          <div className="autofix-section">
            <h4 className="section-title">
              Auto-Fix History
              <span className="fix-count">{result.auto_fix_attempts.length} attempt(s)</span>
            </h4>

            {result.auto_fix_attempts.map((attempt, idx) => {
              const isExpanded = expandedFixes[`${activePair}-${idx}`];
              return (
                <div key={idx} className={`autofix-attempt ${attempt.passed ? 'passed' : 'failed'}`}>
                  <div
                    className="attempt-header"
                    onClick={() => toggleFix(activePair, idx)}
                  >
                    <div className="attempt-info">
                      <span className={`attempt-status ${attempt.passed ? 'status-pass' : 'status-fail'}`}>
                        {attempt.passed ? '✓' : '✗'}
                      </span>
                      <span className="attempt-number">Auto-Fix Attempt {attempt.attempt_number}</span>
                    </div>
                    <div className="attempt-scores">
                      <span className="score-label">Critic:</span>
                      <span className={`score-value ${attempt.critic_recheck_score >= 0.95 ? 'score-high' : 'score-medium'}`}>
                        {(attempt.critic_recheck_score * 100).toFixed(0)}%
                      </span>
                      <span className="score-label">Re-test:</span>
                      <span className={`score-value ${attempt.retest_score >= 0.95 ? 'score-high' : 'score-medium'}`}>
                        {(attempt.retest_score * 100).toFixed(0)}%
                      </span>
                      <span className={`expand-icon ${isExpanded ? 'expanded' : ''}`}>
                        {isExpanded ? '▼' : '▶'}
                      </span>
                    </div>
                  </div>

                  {isExpanded && (
                    <div className="attempt-details">
                      <div className="detail-block">
                        <h5>Tester Feedback that Triggered Fix</h5>
                        <div className="detail-content markdown-content">
                          <ReactMarkdown>{attempt.tester_feedback}</ReactMarkdown>
                        </div>
                      </div>
                      <div className="detail-block">
                        <h5>Creator's Fix</h5>
                        <div className="detail-content markdown-content">
                          <ReactMarkdown>{attempt.creator_fix}</ReactMarkdown>
                        </div>
                      </div>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Final Validated Response */}
        <div className="final-validated">
          <h4 className="section-title">
            Final Validated Response
            <span className={`final-badge ${result.passed ? 'passed' : 'not-passed'}`}>
              {result.passed ? 'VALIDATED' : 'NOT VALIDATED'}
            </span>
          </h4>
          <div className="final-response markdown-content">
            <ReactMarkdown>{result.final_response}</ReactMarkdown>
          </div>
        </div>
      </div>
    </div>
  );
}
