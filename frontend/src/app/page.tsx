"use client";

import { useState, useRef, useEffect } from "react";
import ChatMessage from "@/components/ChatMessage";
import ExecutionVisualizer from "@/components/ExecutionVisualizer";
import UploadPanel from "@/components/UploadPanel";

const API_BASE = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000/api/v1";

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  reflection?: ReflectionResult;
  isStreaming?: boolean;
}

interface Citation {
  claim: string;
  source_chunk_id: string;
  doc_id: string;
  source_text: string;
  confidence: number;
}

interface ReflectionResult {
  confidence_score: number;
  is_faithful: boolean;
  flagged_claims: { claim: string; reason: string }[];
  summary: string;
}

interface ExecutionStep {
  step_id: string;
  action: string;
  status: string;
}

export default function HomePage() {
  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [isStreaming, setIsStreaming] = useState(false);
  const [execSteps, setExecSteps] = useState<ExecutionStep[]>([]);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const [uploadOpen, setUploadOpen] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  const handleSend = async () => {
    if (!input.trim() || isStreaming) return;

    const userMessage: Message = {
      id: crypto.randomUUID(),
      role: "user",
      content: input.trim(),
    };

    const assistantMessageId = crypto.randomUUID();
    const assistantMessage: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMessage, assistantMessage]);
    setInput("");
    setIsStreaming(true);
    setExecSteps([
      { step_id: "step_planning", action: "rag_search", status: "running" },
    ]);

    try {
      const res = await fetch(`${API_BASE}/chat/stream`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: userMessage.content }),
      });

      if (!res.ok || !res.body) {
        throw new Error(`API error: ${res.status}`);
      }

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n\n");
        buffer = lines.pop() || "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const data = line.slice(6).trim();
          if (data === "[DONE]") break;

          try {
            const event = JSON.parse(data);

            if (event.type === "token") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? { ...m, content: m.content + event.content }
                    : m
                )
              );
            } else if (event.type === "metadata") {
              setMessages((prev) =>
                prev.map((m) =>
                  m.id === assistantMessageId
                    ? {
                        ...m,
                        citations: event.citations,
                        reflection: event.reflection,
                        isStreaming: false,
                      }
                    : m
                )
              );
              setExecSteps([
                { step_id: "step_1", action: "rag_search", status: "completed" },
                { step_id: "step_2", action: "direct_llm", status: "completed" },
                { step_id: "step_3", action: "reflection", status: "completed" },
              ]);
            } else if (event.type === "error") {
              throw new Error(event.message);
            }
          } catch {
            // Skip malformed events
          }
        }
      }
    } catch (err: unknown) {
      const errorMsg = err instanceof Error ? err.message : "Unknown error";
      setMessages((prev) =>
        prev.map((m) =>
          m.id === assistantMessageId
            ? {
                ...m,
                content: `⚠️ Error: ${errorMsg}`,
                isStreaming: false,
              }
            : m
        )
      );
      setExecSteps([{ step_id: "step_err", action: "error", status: "failed" }]);
    } finally {
      setIsStreaming(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div style={{ display: "flex", height: "100vh", background: "var(--nm-bg-primary)" }}>
      {/* Left Sidebar */}
      <aside
        style={{
          width: "260px",
          borderRight: "1px solid var(--nm-border)",
          padding: "24px 16px",
          display: "flex",
          flexDirection: "column",
          gap: "16px",
          background: "var(--nm-bg-secondary)",
        }}
      >
        {/* Logo */}
        <div style={{ padding: "8px 0 16px" }}>
          <h1
            style={{
              fontSize: "1.4rem",
              fontWeight: 700,
              background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
              WebkitBackgroundClip: "text",
              WebkitTextFillColor: "transparent",
              margin: 0,
            }}
          >
            ⬡ NeuroMesh
          </h1>
          <p style={{ color: "var(--nm-text-muted)", fontSize: "0.75rem", margin: "4px 0 0" }}>
            Adaptive AI Orchestration
          </p>
        </div>

        {/* Nav Buttons */}
        <button
          onClick={() => setUploadOpen(!uploadOpen)}
          style={{
            background: uploadOpen ? "var(--nm-bg-card)" : "transparent",
            border: "1px solid var(--nm-border)",
            borderRadius: "10px",
            padding: "10px 14px",
            color: "var(--nm-text-primary)",
            cursor: "pointer",
            textAlign: "left",
            fontSize: "0.875rem",
            fontWeight: 500,
          }}
        >
          📄 Upload Documents
        </button>

        <button
          onClick={() => setSidebarOpen(!sidebarOpen)}
          style={{
            background: sidebarOpen ? "var(--nm-bg-card)" : "transparent",
            border: "1px solid var(--nm-border)",
            borderRadius: "10px",
            padding: "10px 14px",
            color: "var(--nm-text-primary)",
            cursor: "pointer",
            textAlign: "left",
            fontSize: "0.875rem",
            fontWeight: 500,
          }}
        >
          🔍 Execution Path
        </button>

        <div style={{ flex: 1 }} />

        {/* System Status */}
        <div
          style={{
            background: "var(--nm-bg-card)",
            border: "1px solid var(--nm-border)",
            borderRadius: "10px",
            padding: "12px",
            fontSize: "0.75rem",
          }}
        >
          <p style={{ color: "var(--nm-text-muted)", margin: "0 0 6px", fontWeight: 600, textTransform: "uppercase", letterSpacing: "0.5px" }}>
            System
          </p>
          <div style={{ display: "flex", alignItems: "center", gap: "6px", color: "var(--nm-accent-emerald)" }}>
            <span style={{ width: "6px", height: "6px", borderRadius: "50%", background: "currentColor", display: "inline-block" }} />
            API Connected
          </div>
        </div>
      </aside>

      {/* Main Chat Area */}
      <main style={{ flex: 1, display: "flex", flexDirection: "column", overflow: "hidden" }}>
        {/* Upload Panel */}
        {uploadOpen && (
          <div
            style={{
              borderBottom: "1px solid var(--nm-border)",
              background: "var(--nm-bg-secondary)",
            }}
          >
            <UploadPanel apiBase={API_BASE} />
          </div>
        )}

        {/* Messages */}
        <div style={{ flex: 1, overflowY: "auto", padding: "24px" }}>
          {messages.length === 0 && (
            <div
              style={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "center",
                justifyContent: "center",
                gap: "12px",
                color: "var(--nm-text-muted)",
              }}
            >
              <span style={{ fontSize: "3rem" }}>⬡</span>
              <h2
                style={{
                  fontSize: "1.5rem",
                  fontWeight: 700,
                  background: "linear-gradient(135deg, #8b5cf6, #06b6d4)",
                  WebkitBackgroundClip: "text",
                  WebkitTextFillColor: "transparent",
                  margin: 0,
                }}
              >
                NeuroMesh
              </h2>
              <p style={{ margin: 0, textAlign: "center", maxWidth: "360px" }}>
                Ask anything. NeuroMesh will plan, retrieve context, load the right expert, and generate a verified answer.
              </p>
            </div>
          )}

          {messages.map((msg) => (
            <ChatMessage key={msg.id} message={msg} />
          ))}
          <div ref={messagesEndRef} />
        </div>

        {/* Input Bar */}
        <div
          style={{
            borderTop: "1px solid var(--nm-border)",
            padding: "16px 24px",
            background: "var(--nm-bg-secondary)",
          }}
        >
          <div
            style={{
              display: "flex",
              gap: "12px",
              alignItems: "flex-end",
              maxWidth: "900px",
              margin: "0 auto",
            }}
          >
            <textarea
              id="chat-input"
              value={input}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={handleKeyDown}
              placeholder="Ask NeuroMesh anything… (Enter to send, Shift+Enter for newline)"
              rows={1}
              style={{
                flex: 1,
                background: "var(--nm-bg-card)",
                border: "1px solid var(--nm-border)",
                borderRadius: "12px",
                padding: "12px 16px",
                color: "var(--nm-text-primary)",
                fontSize: "0.9rem",
                resize: "none",
                outline: "none",
                fontFamily: "inherit",
                lineHeight: "1.5",
                maxHeight: "120px",
                overflowY: "auto",
              }}
              onFocus={(e) => (e.target.style.borderColor = "var(--nm-accent-primary)")}
              onBlur={(e) => (e.target.style.borderColor = "var(--nm-border)")}
            />
            <button
              id="send-button"
              onClick={handleSend}
              disabled={isStreaming || !input.trim()}
              className="btn-glow"
              style={{ whiteSpace: "nowrap", height: "44px" }}
            >
              {isStreaming ? "…" : "Send ↑"}
            </button>
          </div>
        </div>
      </main>

      {/* Execution Visualizer Sidebar */}
      {sidebarOpen && (
        <aside
          style={{
            width: "320px",
            borderLeft: "1px solid var(--nm-border)",
            background: "var(--nm-bg-secondary)",
            overflowY: "auto",
          }}
        >
          <ExecutionVisualizer steps={execSteps} />
        </aside>
      )}
    </div>
  );
}
