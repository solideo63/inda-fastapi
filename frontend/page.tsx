// app/page.tsx - Production Version
"use client";

import { useState, useRef, useEffect } from "react";

interface Source {
  title: string;
  abstract: string;
  link: string;
  release_date: string;
  similarity: number;
}

interface ChatMessage {
  role: "user" | "assistant";
  content: string;
  sources?: Source[];
  timestamp: Date;
  model?: string;
}

const ChatInterface = () => {
  const [question, setQuestion] = useState("");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [selectedModel, setSelectedModel] = useState<"groq" | "gemini">("groq");
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Production API URL (HuggingFace Spaces)
  const API_URL = process.env.NEXT_PUBLIC_API_URL || "https://jernihh-magangchatbot-ai.hf.space/api/chat";


  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!question.trim()) return;

    const userMessage: ChatMessage = {
      role: "user",
      content: question,
      timestamp: new Date(),
    };

    setMessages((prev) => [...prev, userMessage]);
    setQuestion("");
    setLoading(true);
    setError(null);

    try {
      const res = await fetch(`${API_URL}/api/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          question: question,
          use_rag: true,
          model: selectedModel,
        }),
      });

      if (!res.ok) {
        throw new Error(`API Error: ${res.status}`);
      }

      const data = await res.json();

      const assistantMessage: ChatMessage = {
        role: "assistant",
        content: data.answer,
        sources: data.sources || [],
        timestamp: new Date(),
        model: data.metadata?.model || selectedModel,
      };

      setMessages((prev) => [...prev, assistantMessage]);
    } catch (err) {
      const errorMessage = err instanceof Error ? err.message : "Terjadi kesalahan";
      setError(errorMessage);

      const errorChatMessage: ChatMessage = {
        role: "assistant",
        content: `Maaf, terjadi kesalahan: ${errorMessage}`,
        timestamp: new Date(),
      };
      setMessages((prev) => [...prev, errorChatMessage]);
    } finally {
      setLoading(false);
    }
  };

  const quickQuestions = [
    "Data inflasi Sumatera Utara terbaru",
    "Tingkat kemiskinan di Sumut",
    "Pertumbuhan ekonomi Sumatera Utara",
    "Data pengangguran Sumut",
  ];

  return (
    <div className="flex h-screen bg-gray-50">
      {/* Sidebar */}
      <div className="w-64 bg-gray-800 text-white p-4 flex flex-col">
        <h2 className="text-xl font-bold mb-2">🤖 INDA</h2>
        <p className="text-xs text-gray-300 mb-4">
          Intelligent Data Assistant - BPS Sumatera Utara
        </p>

        {/* Model Selection */}
        <div className="mb-4">
          <p className="text-xs text-gray-400 mb-2">🧠 Pilih Model:</p>
          <div className="space-y-2">
            <button
              onClick={() => setSelectedModel("groq")}
              disabled={loading}
              className={`w-full text-left text-xs rounded p-2 transition-colors ${selectedModel === "groq"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 hover:bg-gray-600"
                }`}
            >
              ⚡ Llama 3.1 8B (Groq)
            </button>
            <button
              onClick={() => setSelectedModel("gemini")}
              disabled={loading}
              className={`w-full text-left text-xs rounded p-2 transition-colors ${selectedModel === "gemini"
                  ? "bg-blue-600 text-white"
                  : "bg-gray-700 hover:bg-gray-600"
                }`}
            >
              ✨ Gemini 2.0 Flash
            </button>
          </div>
        </div>

        {/* Quick Questions */}
        <div className="mb-4">
          <p className="text-xs text-gray-400 mb-2">💡 Pertanyaan Cepat:</p>
          <div className="space-y-2">
            {quickQuestions.map((q, idx) => (
              <button
                key={idx}
                onClick={() => setQuestion(q)}
                className="w-full text-left text-xs bg-gray-700 hover:bg-gray-600 rounded p-2 transition-colors"
                disabled={loading}
              >
                {q}
              </button>
            ))}
          </div>
        </div>

        <div className="flex-1" />

        {/* Info */}
        <div className="text-xs text-gray-400 mt-4 border-t border-gray-700 pt-4">
          <p className="mb-2">🔧 Tech Stack:</p>
          <ul className="list-disc list-inside space-y-1">
            <li>LangChain RAG</li>
            <li>Qdrant Cloud</li>
            <li>FastAPI</li>
            <li>HuggingFace</li>
          </ul>
        </div>

        <div className="text-xs text-gray-400 mt-4 border-t border-gray-700 pt-4">
          ⚠️ INDA dapat membuat kesalahan. Selalu verifikasi dengan{" "}
          <a
            href="https://sumut.bps.go.id"
            target="_blank"
            rel="noopener noreferrer"
            className="text-blue-400 hover:underline"
          >
            Website BPS
          </a>
        </div>
      </div>

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <div className="bg-white border-b px-6 py-4 shadow-sm">
          <h1 className="text-xl font-semibold text-gray-800">
            Chat dengan INDA
          </h1>
          <p className="text-sm text-gray-500">
            Tanyakan tentang data statistik BPS Sumatera Utara •{" "}
            <span className="text-blue-600">
              {selectedModel === "groq" ? "Llama 3.1 8B" : "Gemini 2.0 Flash"}
            </span>
          </p>
        </div>

        {/* Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && (
            <div className="text-center text-gray-500 mt-20">
              <div className="text-6xl mb-4">💬</div>
              <h3 className="text-xl font-semibold mb-2">
                Mulai Percakapan
              </h3>
              <p className="text-sm mb-4">
                Tanyakan tentang data inflasi, ekonomi, atau statistik lainnya
              </p>
            </div>
          )}

          {messages.map((msg, idx) => (
            <div
              key={idx}
              className={`flex ${msg.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-3xl rounded-lg p-4 ${msg.role === "user"
                    ? "bg-blue-500 text-white"
                    : "bg-white border shadow-sm"
                  }`}
              >
                <div
                  className={`text-sm whitespace-pre-wrap ${msg.role === "user" ? "text-white" : "text-gray-800"
                    }`}
                  dangerouslySetInnerHTML={{ __html: msg.content }}
                />

                <div
                  className={`text-xs mt-2 flex justify-between items-center ${msg.role === "user" ? "text-blue-100" : "text-gray-400"
                    }`}
                >
                  <span>
                    {msg.timestamp.toLocaleTimeString("id-ID", {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                  {msg.model && (
                    <span className="ml-2 text-xs">
                      {msg.model === "groq" ? "⚡ Groq" : "✨ Gemini"}
                    </span>
                  )}
                </div>

                {msg.sources && msg.sources.length > 0 && (
                  <div className="mt-4 border-t pt-3">
                    <p className="text-xs font-semibold text-gray-600 mb-2">
                      📚 Sumber Referensi ({msg.sources.length}):
                    </p>
                    <div className="space-y-2">
                      {msg.sources.map((source, sidx) => (
                        <div
                          key={sidx}
                          className="bg-gray-50 rounded p-3 text-xs hover:bg-gray-100"
                        >
                          <div className="font-semibold text-gray-700 mb-1">
                            {sidx + 1}. {source.title}
                          </div>
                          <div className="text-gray-600 text-xs mb-2">
                            {source.abstract.substring(0, 150)}...
                          </div>
                          <div className="flex justify-between items-center">
                            <span className="text-gray-500">
                              📅 {source.release_date}
                            </span>
                          </div>
                          {source.link && (
                            <a
                              href={source.link}
                              target="_blank"
                              rel="noopener noreferrer"
                              className="inline-block mt-2 text-blue-500 hover:underline"
                            >
                              📄 Lihat Dokumen →
                            </a>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </div>
            </div>
          ))}

          {loading && (
            <div className="flex justify-start">
              <div className="bg-white border shadow-sm rounded-lg p-4">
                <div className="flex items-center space-x-2">
                  <div className="animate-bounce">🤖</div>
                  <div className="text-sm text-gray-600">
                    INDA sedang menganalisis...
                  </div>
                </div>
              </div>
            </div>
          )}

          {error && (
            <div className="bg-red-50 border border-red-200 rounded-lg p-4 text-red-700 text-sm">
              <div className="flex items-start">
                <span className="mr-2">❌</span>
                <div>
                  <p className="font-semibold mb-1">Error:</p>
                  <p>{error}</p>
                </div>
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white border-t p-4">
          <form onSubmit={handleSubmit} className="flex gap-2 max-w-4xl mx-auto">
            <input
              type="text"
              placeholder="Ketik pertanyaan Anda di sini..."
              value={question}
              onChange={(e) => setQuestion(e.target.value)}
              className="flex-1 border border-gray-300 rounded-lg px-4 py-3 focus:outline-none focus:ring-2 focus:ring-blue-400 text-gray-800"
              disabled={loading}
              required
            />
            <button
              type="submit"
              disabled={loading || !question.trim()}
              className={`px-6 py-3 rounded-lg font-medium transition-all ${loading || !question.trim()
                  ? "bg-gray-300 text-gray-500 cursor-not-allowed"
                  : "bg-blue-500 hover:bg-blue-600 text-white hover:shadow-lg"
                }`}
            >
              {loading ? "⏳" : "📤"}
            </button>
          </form>
          <p className="text-xs text-center text-gray-400 mt-2">
            Press Enter to send • INDA v1.0 • Powered by LangChain
          </p>
        </div>
      </div>
    </div>
  );
};

export default ChatInterface;