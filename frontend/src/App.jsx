import { useState, useEffect } from 'react';
import Sidebar from './components/Sidebar';
import ChatInterface from './components/ChatInterface';
import { api } from './api';
import './App.css';

function App() {
  const [conversations, setConversations] = useState([]);
  const [currentConversationId, setCurrentConversationId] = useState(null);
  const [currentConversation, setCurrentConversation] = useState(null);
  const [isLoading, setIsLoading] = useState(false);
  const [useEnhancedMode, setUseEnhancedMode] = useState(true); // Default to enhanced mode

  // Load conversations on mount
  useEffect(() => {
    loadConversations();
  }, []);

  // Load conversation details when selected
  useEffect(() => {
    if (currentConversationId) {
      loadConversation(currentConversationId);
    }
  }, [currentConversationId]);

  const loadConversations = async () => {
    try {
      const convs = await api.listConversations();
      setConversations(convs);
    } catch (error) {
      console.error('Failed to load conversations:', error);
    }
  };

  const loadConversation = async (id) => {
    try {
      const conv = await api.getConversation(id);
      setCurrentConversation(conv);
    } catch (error) {
      console.error('Failed to load conversation:', error);
    }
  };

  const handleNewConversation = async () => {
    try {
      const newConv = await api.createConversation();
      setConversations([
        { id: newConv.id, created_at: newConv.created_at, message_count: 0 },
        ...conversations,
      ]);
      setCurrentConversationId(newConv.id);
    } catch (error) {
      console.error('Failed to create conversation:', error);
    }
  };

  const handleSelectConversation = (id) => {
    setCurrentConversationId(id);
  };

  const handleSendMessage = async (content) => {
    if (!currentConversationId) return;

    setIsLoading(true);
    try {
      // Optimistically add user message to UI
      const userMessage = { role: 'user', content };
      setCurrentConversation((prev) => ({
        ...prev,
        messages: [...prev.messages, userMessage],
      }));

      if (useEnhancedMode) {
        // Enhanced mode with Agent Pairs
        await handleEnhancedMessage(content);
      } else {
        // Traditional 3-stage mode
        await handleTraditionalMessage(content);
      }
    } catch (error) {
      console.error('Failed to send message:', error);
      // Remove optimistic messages on error
      setCurrentConversation((prev) => ({
        ...prev,
        messages: prev.messages.slice(0, -2),
      }));
      setIsLoading(false);
    }
  };

  // Traditional 3-stage message handler
  const handleTraditionalMessage = async (content) => {
    // Create a partial assistant message that will be updated progressively
    const assistantMessage = {
      role: 'assistant',
      stage1: null,
      stage2: null,
      stage3: null,
      metadata: null,
      loading: {
        stage1: false,
        stage2: false,
        stage3: false,
      },
    };

    // Add the partial assistant message
    setCurrentConversation((prev) => ({
      ...prev,
      messages: [...prev.messages, assistantMessage],
    }));

    // Send message with streaming
    await api.sendMessageStream(currentConversationId, content, (eventType, event) => {
      switch (eventType) {
        case 'stage1_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage1 = true;
            return { ...prev, messages };
          });
          break;

        case 'stage1_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage1 = event.data;
            lastMsg.loading.stage1 = false;
            return { ...prev, messages };
          });
          break;

        case 'stage2_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage2 = true;
            return { ...prev, messages };
          });
          break;

        case 'stage2_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage2 = event.data;
            lastMsg.metadata = event.metadata;
            lastMsg.loading.stage2 = false;
            return { ...prev, messages };
          });
          break;

        case 'stage3_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage3 = true;
            return { ...prev, messages };
          });
          break;

        case 'stage3_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.stage3 = event.data;
            lastMsg.loading.stage3 = false;
            return { ...prev, messages };
          });
          break;

        case 'title_complete':
          loadConversations();
          break;

        case 'complete':
          loadConversations();
          setIsLoading(false);
          break;

        case 'error':
          console.error('Stream error:', event.message);
          setIsLoading(false);
          break;

        default:
          console.log('Unknown event type:', eventType);
      }
    });
  };

  // Enhanced 4-stage message handler with Agent Pairs
  const handleEnhancedMessage = async (content) => {
    // Create a partial assistant message for enhanced mode
    const assistantMessage = {
      role: 'assistant',
      stage_pairs: null,
      stage_validation: null,
      stage_rankings: null,
      stage_synthesis: null,
      metadata: null,
      loading: {
        pairs: false,
        validation: false,
        stage2: false,
        stage3: false,
        pairs_progress: null,
      },
    };

    // Add the partial assistant message
    setCurrentConversation((prev) => ({
      ...prev,
      messages: [...prev.messages, assistantMessage],
    }));

    // Send message with enhanced streaming
    await api.sendMessageEnhancedStream(currentConversationId, content, (eventType, event) => {
      switch (eventType) {
        case 'pairs_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.pairs = true;
            lastMsg.loading.pairs_progress = `Starting ${event.data?.pairs?.length || 0} agent pairs...`;
            return { ...prev, messages };
          });
          break;

        case 'pair_iteration':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            const { pair, iteration, score, status } = event.data || {};
            lastMsg.loading.pairs_progress = `${pair}: Iteration ${iteration} (score: ${(score * 100).toFixed(0)}%) - ${status}`;
            return { ...prev, messages };
          });
          break;

        case 'pairs_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.pairs = false;
            lastMsg.loading.pairs_progress = null;
            return { ...prev, messages };
          });
          break;

        case 'validation_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.validation = true;
            return { ...prev, messages };
          });
          break;

        case 'validation_progress':
          // Could add detailed validation progress here
          break;

        case 'validation_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.validation = false;
            return { ...prev, messages };
          });
          break;

        case 'rankings_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage2 = true;
            return { ...prev, messages };
          });
          break;

        case 'rankings_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage2 = false;
            return { ...prev, messages };
          });
          break;

        case 'synthesis_start':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage3 = true;
            return { ...prev, messages };
          });
          break;

        case 'synthesis_complete':
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            lastMsg.loading.stage3 = false;
            return { ...prev, messages };
          });
          break;

        case 'title_complete':
          loadConversations();
          break;

        case 'complete':
          // Update with full result data
          setCurrentConversation((prev) => {
            const messages = [...prev.messages];
            const lastMsg = messages[messages.length - 1];
            if (event.data) {
              lastMsg.stage_pairs = event.data.stage_pairs;
              lastMsg.stage_validation = event.data.stage_validation;
              lastMsg.stage_rankings = event.data.stage_rankings;
              lastMsg.stage_synthesis = event.data.stage_synthesis;
              lastMsg.metadata = event.data.metadata;
            }
            return { ...prev, messages };
          });
          loadConversations();
          setIsLoading(false);
          break;

        case 'error':
          console.error('Enhanced stream error:', event.message);
          setIsLoading(false);
          break;

        default:
          console.log('Unknown enhanced event type:', eventType);
      }
    });
  };

  return (
    <div className="app">
      <Sidebar
        conversations={conversations}
        currentConversationId={currentConversationId}
        onSelectConversation={handleSelectConversation}
        onNewConversation={handleNewConversation}
      />
      <div className="main-content">
        <div className="mode-toggle">
          <label className="toggle-label">
            <input
              type="checkbox"
              checked={useEnhancedMode}
              onChange={(e) => setUseEnhancedMode(e.target.checked)}
              disabled={isLoading}
            />
            <span className="toggle-slider"></span>
            <span className="toggle-text">
              {useEnhancedMode ? 'Enhanced Mode (Agent Pairs)' : 'Traditional Mode (3-Stage)'}
            </span>
          </label>
        </div>
        <ChatInterface
          conversation={currentConversation}
          onSendMessage={handleSendMessage}
          isLoading={isLoading}
        />
      </div>
    </div>
  );
}

export default App;
