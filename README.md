# 🏦 FedRAG Engine: Deterministic Macroeconomic Signal Extraction

**Live Demo:** [Click here to view the live app on Streamlit Cloud](https://ayush5786-fedrag-engine-fed-app-5asgp0.streamlit.app/)

![App Screenshot](FedRAG-Engine.png)

## Overview

The FedRAG Engine is an automated, real-time extraction pipeline designed for quantitative finance. It dynamically scrapes live XML feeds from the U.S. Federal Reserve, builds an in-memory vector database, and utilizes the Groq Llama-3.3 70B model to extract macroeconomic signals (Hawkish, Dovish, Neutral).

To ensure high-fidelity context, the system features a dynamic slider allowing users to scale the active context pool (1 to 10 speeches) using a custom "Find-Exactly-N" validation filter. Most importantly, it features a strict **Python Set-Theory safety net** to mathematically detect and prevent LLM hallucinations, completely replacing unreliable LLM confidence scores.

## 🛠️ Architecture & Tech Stack

* **Modular Data Ingestion:** A dynamic "Fail-Open" pipeline using `feedparser` and `BeautifulSoup4`. It continuously polls XML metadata and applies a Python Keyword Bouncer to drop irrelevant speeches until the exact user-defined context target is met.
* **Semantic Chunking:** Custom double-newline (`\n\n`) aggregation (max 300 words) to preserve paragraph logic, replacing naïve character-splitting.
* **Vector Database:** `SentenceTransformers` (`all-MiniLM-L6-v2`) and `FAISS` (L2 distance) for localized, in-memory embedding storage.
* **LLM Engine:** `llama-3.3-70b-versatile` running via the **Groq API** for ultra-low latency generation (T=0.1).
* **Frontend UI:** Built dynamically with `Streamlit`, featuring `@st.cache_resource` and `@st.cache_data` to cache the FAISS index, **cutting query latency by 86%** (~15s down to ~2s). It includes a Transparency UI exposing the exact active speech links and source-evidence chunks.

## 🛡️ Hallucination Prevention (The Safety Net)

Large Language Models are prone to hallucinating financial quotes. To make this production-ready, I engineered a deterministic validation layer that removes the burden of trust from the LLM:

1. **Strict Context Boundaries:** The LLM evaluates retrieved chunks against a strict macroeconomic grading rubric. If the query is unrelated to the context, the system forces a graceful failure ("Out of Scope") instead of guessing.
2. **Deterministic Math over Vibes:** The LLM is restricted from generating its own "confidence scores." Instead, it extracts a verbatim quote to justify its signal.
3. **Set-Theory Interception:** The Python backend intercepts the quote, strips punctuation, converts the strings into Sets, and calculates a fuzzy-matching intersection against the raw FAISS chunks.
4. **The Block:** If the mathematical overlap is below 90%, the system overrides the LLM, blocks the output entirely, and throws a deterministic hallucination error to the user.

## 🚀 How to Run Locally

1. Clone this repository.
2. Install the requirements: `pip install -r requirements.txt`
3. Add your Groq API key to `.streamlit/secrets.toml`
4. Run the app: `streamlit run app.py`
