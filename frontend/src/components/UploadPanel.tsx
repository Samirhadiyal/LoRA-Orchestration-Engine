"use client";

import { useState, useRef } from "react";

interface Props {
  apiBase: string;
}

interface UploadResult {
  status: string;
  document_id: number;
  filename: string;
  total_chunks_created: number;
}

export default function UploadPanel({ apiBase }: Props) {
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState<UploadResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [dragOver, setDragOver] = useState(false);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFile = async (file: File) => {
    if (!file) return;
    const allowed = ["application/pdf", "text/plain", "text/markdown", "application/vnd.openxmlformats-officedocument.wordprocessingml.document"];
    if (!allowed.includes(file.type) && !file.name.endsWith(".md")) {
      setError("Unsupported file type. Please upload PDF, DOCX, TXT, or MD.");
      return;
    }

    setUploading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append("file", file);

    try {
      const res = await fetch(`${apiBase}/documents/upload`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();

      if (!res.ok) {
        throw new Error(data.detail || `Upload failed: ${res.status}`);
      }

      setResult(data);
    } catch (err: unknown) {
      setError(err instanceof Error ? err.message : "Upload failed.");
    } finally {
      setUploading(false);
    }
  };

  const onDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  return (
    <div style={{ padding: "20px 24px" }}>
      <h3
        style={{
          fontSize: "0.875rem",
          fontWeight: 700,
          color: "var(--nm-text-secondary)",
          textTransform: "uppercase",
          letterSpacing: "1px",
          margin: "0 0 16px",
        }}
      >
        📄 Upload to Knowledge Base
      </h3>

      {/* Drop zone */}
      <div
        id="upload-dropzone"
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={onDrop}
        onClick={() => fileInputRef.current?.click()}
        style={{
          border: `2px dashed ${dragOver ? "var(--nm-accent-primary)" : "var(--nm-border)"}`,
          borderRadius: "12px",
          padding: "24px",
          textAlign: "center",
          cursor: "pointer",
          background: dragOver ? "var(--nm-accent-glow)" : "transparent",
          transition: "all 0.2s ease",
          color: "var(--nm-text-muted)",
          fontSize: "0.875rem",
        }}
      >
        {uploading ? (
          <span style={{ color: "var(--nm-accent-primary)" }}>⟳ Uploading and indexing…</span>
        ) : (
          <>
            <p style={{ margin: "0 0 4px" }}>
              Drop a file here or <span style={{ color: "var(--nm-accent-cyan)", textDecoration: "underline" }}>browse</span>
            </p>
            <p style={{ margin: 0, fontSize: "0.7rem" }}>PDF · DOCX · TXT · Markdown</p>
          </>
        )}
        <input
          ref={fileInputRef}
          id="file-upload-input"
          type="file"
          accept=".pdf,.docx,.txt,.md"
          style={{ display: "none" }}
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleFile(file);
          }}
        />
      </div>

      {/* Success Result */}
      {result && (
        <div
          style={{
            marginTop: "12px",
            padding: "12px 16px",
            background: "rgba(16, 185, 129, 0.1)",
            border: "1px solid rgba(16, 185, 129, 0.3)",
            borderRadius: "10px",
            fontSize: "0.8rem",
          }}
        >
          <p style={{ margin: "0 0 4px", color: "var(--nm-accent-emerald)", fontWeight: 600 }}>
            ✓ Upload successful
          </p>
          <p style={{ margin: 0, color: "var(--nm-text-secondary)" }}>
            <strong>{result.filename}</strong> — {result.total_chunks_created} chunks indexed
          </p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div
          style={{
            marginTop: "12px",
            padding: "12px 16px",
            background: "rgba(244, 63, 94, 0.1)",
            border: "1px solid rgba(244, 63, 94, 0.3)",
            borderRadius: "10px",
            fontSize: "0.8rem",
            color: "var(--nm-accent-rose)",
          }}
        >
          ⚠ {error}
        </div>
      )}
    </div>
  );
}
