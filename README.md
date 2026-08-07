<div align="center">

#  LoRA Orchestration Engine

*Self-orchestrating AI backend*

![Status](https://img.shields.io/badge/status-in%20development-yellow?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=flat-square&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white)
![Qdrant](https://img.shields.io/badge/Qdrant-DC244C?style=flat-square&logo=qdrant&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat-square&logo=docker&logoColor=white)

</div>

---

```mermaid
%%{init: {'theme':'base', 'themeVariables': {
  'primaryColor':'#ede9fe','primaryBorderColor':'#a78bfa','primaryTextColor':'#1e1b4b',
  'lineColor':'#64748b','fontFamily':'Verdana'
}}}%%
flowchart TB
    U(["👤 Query"]) --> P["🧭 Planner"] --> R["🔀 Router"]
    R --> H["📚 Hybrid Retrieval"]
    R --> T["🛠️ Tool Router"]
    R --> L["🧩 LoRA Manager"]
    H --> C["🗜️ Context Optimizer"]
    T --> C
    L --> C
    C --> G["🤖 Reasoning Engine"]
    G --> RF["🔍 Reflection"] --> CI["🔗 Citations"] --> S["📡 Stream"] --> F(["💬 Chat UI"])

    classDef done fill:#dcfce7,stroke:#4ade80,color:#14532d;
    classDef progress fill:#fef3c7,stroke:#fbbf24,color:#78350f;
    classDef pending fill:#fee2e2,stroke:#f87171,color:#7f1d1d;
    class H,T progress
    class L,C,P,R,G,RF,CI,S,F pending
```

