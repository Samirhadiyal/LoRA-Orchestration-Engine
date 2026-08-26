import type { Metadata } from "next";
import "./globals.css";

export const metadata: Metadata = {
  title: "NeuroMesh — Adaptive AI Orchestration",
  description:
    "NeuroMesh dynamically routes queries through specialized LoRA adapters, RAG pipelines, and tool agents for intelligent, grounded responses.",
};

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode;
}>) {
  return (
    <html lang="en">
      <head>
        <link
          href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap"
          rel="stylesheet"
        />
      </head>
      <body>{children}</body>
    </html>
  );
}
