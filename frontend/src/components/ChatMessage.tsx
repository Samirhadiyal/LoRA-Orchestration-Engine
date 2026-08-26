"use client";

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

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  citations?: Citation[];
  reflection?: ReflectionResult;
  isStreaming?: boolean;
}

interface Props {
  message: Message;
}

export default function ChatMessage({ message }: Props) {
  const isUser = message.role === "user";

  return (
    <div
      style={{
        display: "flex",
        flexDirection: "column",
        alignItems: isUser ? "flex-end" : "flex-start",
        marginBottom: "20px",
        maxWidth: "900px",
        margin: "0 auto 20px",
      }}
    >
      {/* Role Label */}
      <span
        style={{
          fontSize: "0.7rem",
          fontWeight: 600,
          color: "var(--nm-text-muted)",
          textTransform: "uppercase",
          letterSpacing: "0.5px",
          marginBottom: "6px",
          paddingLeft: isUser ? 0 : "4px",
          paddingRight: isUser ? "4px" : 0,
          alignSelf: isUser ? "flex-end" : "flex-start",
        }}
      >
        {isUser ? "You" : "⬡ NeuroMesh"}
      </span>

      {/* Bubble */}
      <div
        style={{
          maxWidth: "80%",
          padding: "14px 18px",
          borderRadius: isUser ? "18px 18px 4px 18px" : "4px 18px 18px 18px",
          background: isUser
            ? "linear-gradient(135deg, var(--nm-accent-primary), var(--nm-accent-secondary))"
            : "var(--nm-bg-card)",
          border: isUser ? "none" : "1px solid var(--nm-border)",
          color: "var(--nm-text-primary)",
          fontSize: "0.9rem",
          lineHeight: "1.6",
          whiteSpace: "pre-wrap",
          wordBreak: "break-word",
        }}
      >
        {message.content}

        {/* Streaming indicator */}
        {message.isStreaming && (
          <span style={{ marginLeft: "8px" }}>
            <span className="typing-dot" />
            <span className="typing-dot" />
            <span className="typing-dot" />
          </span>
        )}
      </div>

      {/* Reflection Score */}
      {message.reflection && !message.isStreaming && (
        <div
          style={{
            marginTop: "8px",
            display: "flex",
            alignItems: "center",
            gap: "10px",
            fontSize: "0.75rem",
          }}
        >
          <span
            style={{
              padding: "3px 10px",
              borderRadius: "12px",
              background: message.reflection.is_faithful
                ? "rgba(16, 185, 129, 0.1)"
                : "rgba(244, 63, 94, 0.1)",
              border: `1px solid ${message.reflection.is_faithful ? "rgba(16, 185, 129, 0.3)" : "rgba(244, 63, 94, 0.3)"}`,
              color: message.reflection.is_faithful
                ? "var(--nm-accent-emerald)"
                : "var(--nm-accent-rose)",
              fontWeight: 600,
            }}
          >
            {message.reflection.is_faithful ? "✓ Verified" : "⚠ Unverified"}
          </span>
          <span style={{ color: "var(--nm-text-muted)" }}>
            Confidence: {Math.round(message.reflection.confidence_score * 100)}%
          </span>
        </div>
      )}

      {/* Citations */}
      {message.citations && message.citations.length > 0 && !message.isStreaming && (
        <div
          style={{
            marginTop: "10px",
            width: "100%",
            maxWidth: "80%",
          }}
        >
          <details>
            <summary
              style={{
                cursor: "pointer",
                color: "var(--nm-accent-cyan)",
                fontSize: "0.75rem",
                fontWeight: 600,
                userSelect: "none",
                outline: "none",
              }}
            >
              📎 {message.citations.length} source{message.citations.length !== 1 ? "s" : ""} cited
            </summary>
            <div
              style={{
                marginTop: "8px",
                display: "flex",
                flexDirection: "column",
                gap: "6px",
              }}
            >
              {message.citations.slice(0, 3).map((c, i) => (
                <div
                  key={i}
                  style={{
                    background: "var(--nm-bg-glass)",
                    border: "1px solid var(--nm-border)",
                    borderRadius: "8px",
                    padding: "8px 12px",
                    fontSize: "0.75rem",
                  }}
                >
                  <p style={{ margin: "0 0 4px", color: "var(--nm-text-secondary)", fontStyle: "italic" }}>
                    "{c.claim.substring(0, 120)}{c.claim.length > 120 ? "…" : ""}"
                  </p>
                  <p style={{ margin: 0, color: "var(--nm-text-muted)" }}>
                    {c.source_text.substring(0, 100)}… — <strong>{Math.round(c.confidence * 100)}% match</strong>
                  </p>
                </div>
              ))}
            </div>
          </details>
        </div>
      )}
    </div>
  );
}
