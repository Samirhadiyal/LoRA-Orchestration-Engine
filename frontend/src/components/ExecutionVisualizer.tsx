"use client";

interface ExecutionStep {
  step_id: string;
  action: string;
  status: string;
}

interface Props {
  steps: ExecutionStep[];
}

const ACTION_LABELS: Record<string, { label: string; icon: string }> = {
  rag_search: { label: "RAG Search", icon: "🔍" },
  direct_llm: { label: "LLM Synthesis", icon: "🧠" },
  lora_adapter: { label: "LoRA Expert", icon: "⚡" },
  sql_query: { label: "SQL Query", icon: "🗄️" },
  swarm_delegate: { label: "Swarm Delegate", icon: "🐝" },
  reflection: { label: "Reflection Verification", icon: "✓" },
  error: { label: "Error", icon: "✗" },
};

function StatusBadge({ status }: { status: string }) {
  const classMap: Record<string, string> = {
    completed: "status-completed",
    running: "status-running",
    failed: "status-failed",
    pending: "status-pending",
    in_progress: "status-running",
  };

  const dotColors: Record<string, string> = {
    completed: "var(--nm-accent-emerald)",
    running: "var(--nm-accent-primary)",
    failed: "var(--nm-accent-rose)",
    pending: "var(--nm-text-muted)",
    in_progress: "var(--nm-accent-primary)",
  };

  return (
    <span className={`status-badge ${classMap[status] || "status-pending"}`}>
      <span
        style={{
          width: "6px",
          height: "6px",
          borderRadius: "50%",
          background: dotColors[status] || "var(--nm-text-muted)",
          display: "inline-block",
          animation: status === "running" || status === "in_progress" ? "pulse-glow 2s infinite" : "none",
        }}
      />
      {status.replace("_", " ")}
    </span>
  );
}

export default function ExecutionVisualizer({ steps }: Props) {
  return (
    <div style={{ padding: "20px 16px" }}>
      <h3
        style={{
          fontSize: "0.875rem",
          fontWeight: 700,
          color: "var(--nm-text-secondary)",
          textTransform: "uppercase",
          letterSpacing: "1px",
          margin: "0 0 20px",
        }}
      >
        ⚙ Execution Path
      </h3>

      {steps.length === 0 ? (
        <p style={{ color: "var(--nm-text-muted)", fontSize: "0.8rem", textAlign: "center", paddingTop: "40px" }}>
          Send a message to see the execution plan here.
        </p>
      ) : (
        <div style={{ position: "relative" }}>
          {/* Vertical connector line */}
          <div
            style={{
              position: "absolute",
              left: "19px",
              top: "10px",
              bottom: "10px",
              width: "2px",
              background: "var(--nm-border)",
            }}
          />

          <div style={{ display: "flex", flexDirection: "column", gap: "16px" }}>
            {steps.map((step, idx) => {
              const meta = ACTION_LABELS[step.action] || { label: step.action, icon: "▸" };
              return (
                <div
                  key={step.step_id}
                  style={{
                    display: "flex",
                    gap: "12px",
                    alignItems: "flex-start",
                    position: "relative",
                    zIndex: 1,
                  }}
                >
                  {/* Step circle */}
                  <div
                    style={{
                      width: "38px",
                      height: "38px",
                      borderRadius: "50%",
                      background:
                        step.status === "completed"
                          ? "rgba(16, 185, 129, 0.15)"
                          : step.status === "failed"
                          ? "rgba(244, 63, 94, 0.15)"
                          : "var(--nm-bg-card)",
                      border: `2px solid ${
                        step.status === "completed"
                          ? "var(--nm-accent-emerald)"
                          : step.status === "failed"
                          ? "var(--nm-accent-rose)"
                          : step.status === "running" || step.status === "in_progress"
                          ? "var(--nm-accent-primary)"
                          : "var(--nm-border)"
                      }`,
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "center",
                      fontSize: "1rem",
                      flexShrink: 0,
                      boxShadow:
                        step.status === "running" || step.status === "in_progress"
                          ? "0 0 12px var(--nm-accent-glow)"
                          : "none",
                    }}
                  >
                    {meta.icon}
                  </div>

                  {/* Step details */}
                  <div style={{ flex: 1, paddingTop: "4px" }}>
                    <div
                      style={{
                        display: "flex",
                        justifyContent: "space-between",
                        alignItems: "center",
                        marginBottom: "4px",
                      }}
                    >
                      <span
                        style={{
                          fontSize: "0.85rem",
                          fontWeight: 600,
                          color: "var(--nm-text-primary)",
                        }}
                      >
                        {idx + 1}. {meta.label}
                      </span>
                      <StatusBadge status={step.status} />
                    </div>
                    <p
                      style={{
                        margin: 0,
                        fontSize: "0.72rem",
                        color: "var(--nm-text-muted)",
                        fontFamily: "monospace",
                      }}
                    >
                      {step.step_id}
                    </p>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      )}
    </div>
  );
}
