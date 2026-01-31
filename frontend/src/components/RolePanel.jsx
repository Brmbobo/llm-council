import { useState } from 'react';
import './RolePanel.css';

export default function RolePanel({
  availableRoles,
  activeRoles,
  onRolesChange
}) {
  const [expandedRole, setExpandedRole] = useState(null);

  if (!availableRoles) {
    return null;
  }

  const handleToggleRole = (roleId) => {
    const isActive = activeRoles.some(r => r.role_id === roleId);

    if (isActive) {
      // Remove role
      onRolesChange(activeRoles.filter(r => r.role_id !== roleId));
      if (expandedRole === roleId) {
        setExpandedRole(null);
      }
    } else {
      // Add role with default prompt
      onRolesChange([...activeRoles, { role_id: roleId, custom_prompt: null }]);
    }
  };

  const handleExpandRole = (roleId, e) => {
    e.stopPropagation();
    setExpandedRole(expandedRole === roleId ? null : roleId);
  };

  const handlePromptChange = (roleId, newPrompt) => {
    onRolesChange(
      activeRoles.map(r =>
        r.role_id === roleId
          ? { ...r, custom_prompt: newPrompt || null }
          : r
      )
    );
  };

  const handleResetPrompt = (roleId) => {
    handlePromptChange(roleId, null);
  };

  const getPromptForRole = (roleId) => {
    const activeRole = activeRoles.find(r => r.role_id === roleId);
    return activeRole?.custom_prompt || availableRoles[roleId]?.default_prompt || '';
  };

  const isRoleActive = (roleId) => activeRoles.some(r => r.role_id === roleId);
  const isPromptCustomized = (roleId) => {
    const activeRole = activeRoles.find(r => r.role_id === roleId);
    return activeRole?.custom_prompt !== null && activeRole?.custom_prompt !== undefined;
  };

  return (
    <div className="role-panel">
      <div className="role-panel-header">
        <span className="role-panel-label">Roles</span>
        <span className="role-count">{activeRoles.length} active</span>
      </div>

      <div className="role-chips">
        {Object.entries(availableRoles).map(([roleId, roleData]) => (
          <div key={roleId} className="role-chip-container">
            <button
              className={`role-chip ${isRoleActive(roleId) ? 'active' : ''} ${expandedRole === roleId ? 'expanded' : ''}`}
              onClick={() => handleToggleRole(roleId)}
            >
              <span className="role-name">{roleData.name}</span>
              {isRoleActive(roleId) && (
                <>
                  {isPromptCustomized(roleId) && (
                    <span className="customized-indicator" title="Custom prompt">*</span>
                  )}
                  <button
                    className="role-edit-btn"
                    onClick={(e) => handleExpandRole(roleId, e)}
                    title="Edit prompt"
                  >
                    {expandedRole === roleId ? '▲' : '▼'}
                  </button>
                </>
              )}
            </button>
          </div>
        ))}
      </div>

      {expandedRole && isRoleActive(expandedRole) && (
        <div className="prompt-editor">
          <div className="prompt-editor-header">
            <span className="prompt-editor-title">
              {availableRoles[expandedRole]?.name} Prompt
            </span>
            {isPromptCustomized(expandedRole) && (
              <button
                className="reset-btn"
                onClick={() => handleResetPrompt(expandedRole)}
              >
                Reset to Default
              </button>
            )}
          </div>
          <textarea
            className="prompt-textarea"
            value={getPromptForRole(expandedRole)}
            onChange={(e) => handlePromptChange(expandedRole, e.target.value)}
            placeholder="Enter custom prompt..."
            rows={6}
          />
        </div>
      )}
    </div>
  );
}
