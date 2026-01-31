import { useState } from 'react';
import ReactMarkdown from 'react-markdown';
import './Stage2_5.css';

export default function Stage2_5({ revisions, labelToModel }) {
  const [activeTab, setActiveTab] = useState(0);

  if (!revisions || revisions.length === 0) {
    return null;
  }

  const getModelShortName = (model) => model.split('/')[1] || model;

  return (
    <div className="stage stage2-5">
      <h3 className="stage-title">Stage 2.5: Revisions</h3>
      <p className="stage-description">
        Each model revised their response based on peer feedback.
        Compare original vs revised responses below.
      </p>

      <div className="tabs">
        {revisions.map((rev, index) => (
          <button
            key={index}
            className={`tab ${activeTab === index ? 'active' : ''}`}
            onClick={() => setActiveTab(index)}
          >
            <span className="tab-model">{getModelShortName(rev.model)}</span>
            {rev.role_name && (
              <span className="tab-role-badge">{rev.role_name}</span>
            )}
          </button>
        ))}
      </div>

      <div className="revision-content">
        <div className="revision-meta">
          <span className="model-name">{revisions[activeTab].model}</span>
          {revisions[activeTab].label && (
            <span className="model-label">({revisions[activeTab].label})</span>
          )}
          {revisions[activeTab].role_name && (
            <span className="role-badge">{revisions[activeTab].role_name}</span>
          )}
        </div>

        <div className="comparison-grid">
          <div className="comparison-column original">
            <h4 className="comparison-header">Original Response</h4>
            <div className="comparison-body markdown-content">
              <ReactMarkdown>
                {revisions[activeTab].original_response}
              </ReactMarkdown>
            </div>
          </div>

          <div className="comparison-divider"></div>

          <div className="comparison-column revised">
            <h4 className="comparison-header">Revised Response</h4>
            <div className="comparison-body markdown-content">
              <ReactMarkdown>
                {revisions[activeTab].revised_response}
              </ReactMarkdown>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
