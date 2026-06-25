# 🏦 FedRAG Engine: Autonomous Macroeconomic Intelligence Terminal

**Live Demo:** [Click here to view the live app on Streamlit Cloud](https://ayush5786-fedrag-engine-fed-app-5asgp0.streamlit.app/)

![App Screenshot](FedRAG-Engine.png)
*(Note: The browser view is zoomed out in the screenshot above to capture the complete extraction pipeline, dynamic UI, and evidence expanders in a single frame.)*

## Overview

The FedRAG Engine is an automated retrieval-augmented generation (RAG) platform designed to extract complex policy shifts and structural macroeconomic themes from the U.S. Federal Reserve. Rather than relying on generic speech titles, the engine utilizes a **FAISS-first architecture** to scrape, vectorize, and globally rank full-text transcripts (scaling dynamically from 5 to 50 recent speeches). 

Leveraging the **Groq API** and **Llama-3.3 70B**, the system evaluates the optimal text vectors to extract domain-agnostic insights (Theme, Stance, Rationale, and Verbatim Evidence) formatted as strict, color-coded JSON payloads. 

To ensure enterprise-grade reliability, the application features an autonomous **"Macro Pivot" fallback engine** to gracefully handle out-of-scope queries, and a custom **mathematical token-validation layer** that mathematically guarantees the prevention of LLM hallucinations.

## 🛠️ Architecture & Tech Stack

* **Full-Text Ingestion Pipeline:** Dynamically polls the Federal Reserve XML RSS feed and utilizes `BeautifulSoup4` to bypass uninformative metadata, scraping the complete raw text of the selected speech cohort on demand.
* **Semantic Chunking & Global Retrieval:** Text is aggregated into semantic blocks to preserve paragraph logic. `SentenceTransformers` (`all-MiniLM-L6-v2`) and `FAISS` (L2 distance) generate an in-memory vector database, enabling a global mathematical search across all transcripts simultaneously.
* **Multi-LLM Orchestration & Fallback Evolution:** Powered by the **Groq API** for ultra-low latency. The primary pipeline utilizes `Llama 3.3 (70B)` for deep qualitative reasoning. Following the deprecation of the initial fallback model (`Llama 4 Scout 17B`), the architecture was migrated to `Qwen 3.6 (27B)` to handle 429 rate limits.
    * *Why Qwen 3.6 (27B)?* It serves as the optimal mid-tier bridge. It operates on an isolated rate-limit pool from the primary Llama model, handles structural JSON formatting cleanly, and maintains high-level logical benchmarks required for complex financial extraction.
    * *Why not smaller models (e.g., Llama 8B)?* While 8B models offer massive daily request ceilings on Groq's free tier, they lack the nuanced reasoning capacity needed to accurately parse complex Federal Reserve syntax when substituting for the 70B model.
    * *Why not massive models (e.g., 120B+)?* Larger open-weight models introduce severe token consumption overhead and tighter per-minute caps, risking pipeline stagnation during multi-query lookups.
* **Frontend UI:** Built with `Streamlit`, leveraging `@st.cache_resource` and `@st.cache_data` to persist the FAISS index in memory. This eliminates redundant scraping and embedding overhead, **cutting repeat-query latency by 86%** (~15s down to ~2s).

## 🛡️ Enterprise Guardrails & Hallucination Prevention

Large Language Models are prone to hallucinating financial quotes. This system removes the burden of trust from the LLM via deterministic engineering:

1. **The "Macro Pivot" Fallback:** The LLM is strictly prompted to evaluate the presence of the user's core subject within the vectors. If a query (e.g., "Mars exploration") is mathematically missing from the text, the LLM flags it as "Out of Scope." The Streamlit app intercepts this, blocks standard extraction, and dynamically compiles a 3-bullet Executive Briefing on what central bankers are *actually* prioritizing instead.
2. **Deterministic Math over Vibes:** The LLM is restricted from generating its own "confidence scores." Instead, it must extract a verbatim quote to justify its stance.
3. **Set-Theory Interception:** The Python backend intercepts the generated quote, strips punctuation, converts the strings into Sets, and calculates a fuzzy-matching intersection against the raw source FAISS chunks.
4. **The Block:** If the mathematical overlap is below 90%, the system overrides the LLM, blocks the output entirely, and throws a deterministic hallucination error to the user, guaranteeing absolute data ground truth.

## 🚀 How to Run Locally

1. Clone this repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Add your Groq API key to `.streamlit/secrets.toml`
4. Run the app: `streamlit run fed_app.py`
