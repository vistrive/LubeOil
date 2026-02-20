import { useState, useRef, useEffect } from 'react';
import { Send, Bot, User, Loader2, Settings2 } from 'lucide-react';

const API_BASE = '/api/v1';

interface Message {
  role: 'user' | 'assistant';
  content: string;
  provider?: string;
  model?: string;
  error?: string;
}

const PROVIDERS = [
  { id: 'openai', label: 'ChatGPT' },
  { id: 'anthropic', label: 'Claude' },
  { id: 'ollama', label: 'Llama (Local)' },
  { id: 'sarvam', label: 'Sarvam.ai' },
];

export default function AIChat() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState('');
  const [provider, setProvider] = useState('openai');
  const [loading, setLoading] = useState(false);
  const [showSettings, setShowSettings] = useState(false);
  const messagesEnd = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEnd.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const sendMessage = async () => {
    if (!input.trim() || loading) return;
    const userMsg: Message = { role: 'user', content: input.trim() };
    setMessages((prev) => [...prev, userMsg]);
    setInput('');
    setLoading(true);

    try {
      const resp = await fetch(`${API_BASE}/llm/chat`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ message: userMsg.content, provider }),
      });
      const data = await resp.json();

      const aiMsg: Message = {
        role: 'assistant',
        content: data.content || data.detail || 'No response',
        provider: data.provider,
        model: data.model,
        error: data.error || undefined,
      };
      setMessages((prev) => [...prev, aiMsg]);
    } catch (e: any) {
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: '', error: e.message },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  return (
    <div className="flex flex-col h-[calc(100vh-8rem)]">
      {/* Header */}
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-2xl font-bold text-gray-900">AI Formulation Assistant</h2>
        <button
          onClick={() => setShowSettings(!showSettings)}
          className="btn btn-secondary text-sm"
        >
          <Settings2 className="w-4 h-4" />
          Provider
        </button>
      </div>

      {/* Provider selector */}
      {showSettings && (
        <div className="card mb-4">
          <h4 className="text-sm font-semibold text-gray-700 mb-2">LLM Provider</h4>
          <div className="flex gap-2 flex-wrap">
            {PROVIDERS.map((p) => (
              <button
                key={p.id}
                onClick={() => { setProvider(p.id); setShowSettings(false); }}
                className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                  provider === p.id
                    ? 'bg-blue-600 text-white'
                    : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>
      )}

      {/* Messages */}
      <div className="flex-1 overflow-y-auto space-y-4 mb-4 p-4 bg-gray-50 rounded-lg">
        {messages.length === 0 && (
          <div className="text-center text-gray-400 py-12">
            <Bot className="w-12 h-12 mx-auto mb-3 opacity-50" />
            <p className="text-lg font-medium">Ask about lubricant formulation</p>
            <p className="text-sm mt-1">
              Try: "Suggest a 20W-50 engine oil formulation" or "What base oils work best for gear oils?"
            </p>
          </div>
        )}
        {messages.map((msg, idx) => (
          <div
            key={idx}
            className={`flex gap-3 ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}
          >
            {msg.role === 'assistant' && (
              <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center flex-shrink-0">
                <Bot className="w-4 h-4 text-blue-600" />
              </div>
            )}
            <div
              className={`max-w-[75%] rounded-lg px-4 py-3 ${
                msg.role === 'user'
                  ? 'bg-blue-600 text-white'
                  : msg.error
                  ? 'bg-red-50 text-red-700 border border-red-200'
                  : 'bg-white border border-gray-200'
              }`}
            >
              {msg.error ? (
                <p className="text-sm">Error: {msg.error}</p>
              ) : (
                <div className="text-sm whitespace-pre-wrap">{msg.content}</div>
              )}
              {msg.provider && (
                <p className="text-xs mt-2 opacity-60">
                  {msg.provider} / {msg.model}
                </p>
              )}
            </div>
            {msg.role === 'user' && (
              <div className="w-8 h-8 bg-gray-200 rounded-full flex items-center justify-center flex-shrink-0">
                <User className="w-4 h-4 text-gray-600" />
              </div>
            )}
          </div>
        ))}
        {loading && (
          <div className="flex gap-3">
            <div className="w-8 h-8 bg-blue-100 rounded-full flex items-center justify-center">
              <Bot className="w-4 h-4 text-blue-600" />
            </div>
            <div className="bg-white border border-gray-200 rounded-lg px-4 py-3">
              <Loader2 className="w-4 h-4 animate-spin text-gray-400" />
            </div>
          </div>
        )}
        <div ref={messagesEnd} />
      </div>

      {/* Input */}
      <div className="flex gap-2">
        <textarea
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder="Ask about formulation, viscosity blending, base oils..."
          rows={2}
          className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 resize-none"
        />
        <button
          onClick={sendMessage}
          disabled={loading || !input.trim()}
          className="btn btn-primary px-4 self-end"
        >
          <Send className="w-4 h-4" />
        </button>
      </div>
    </div>
  );
}
