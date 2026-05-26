import streamlit as st
import json
import requests
import feedparser
from bs4 import BeautifulSoup
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import os
import re
from groq import Groq

# --- UI: Page Setup ---
st.set_page_config(page_title="FedRAG Engine", layout="wide")
st.title("🏦 Federal Reserve RAG Pipeline")
st.write("Extracting macroeconomic interest rate signals from live Fed speeches.")

# --- Caching the Heavy Lifting ---

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer('all-MiniLM-L6-v2')

embed_model = load_embedding_model()

@st.cache_resource(ttl=3600, show_spinner="Initializing Vector Database (This only happens once)...")
def build_vector_database():
    # 1. Scrape the Fed RSS Feed
    rss_url = "https://www.federalreserve.gov/feeds/speeches.xml"
    feed = feedparser.parse(rss_url)
    latest_speech_url = feed.entries[0].link
    
    response = requests.get(latest_speech_url)
    soup = BeautifulSoup(response.text, "html.parser")
    article_div = soup.find("div", id="article")
    paragraphs = article_div.find_all("p") if article_div else soup.find_all("p")
    
    raw_text = ""
    for p in paragraphs:
        text = p.get_text().strip()
        if text and "views expressed are my own" not in text.lower():
            raw_text += text + "\n\n"
            
    # 2. Chunk the text
    def chunk_text(text, max_words=300):
        paragraphs = text.split("\n\n")
        chunks, current_chunk, word_count = [], [], 0
        for p in paragraphs:
            words = p.split()
            if word_count + len(words) > max_words and current_chunk:
                chunks.append(" ".join(current_chunk))
                current_chunk, word_count = [], 0
            current_chunk.append(p)
            word_count += len(words)
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    speech_chunks = chunk_text(raw_text)
    
    # 3. Build FAISS Database
    embeddings = embed_model.encode(speech_chunks)
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))
    
    # FIX: We now return the URL as well!
    return speech_chunks, index, latest_speech_url

# FIX: We catch the URL variable here!
speech_chunks, index, latest_speech_url = build_vector_database()

# --- UI: Interactive Elements ---
st.sidebar.header("Configuration")
api_key = st.sidebar.text_input("Enter Groq API Key:", type="password")

st.markdown("### Semantic Search Engine")
user_query = st.text_input("Ask a question about the latest Fed Speech:")

# --- UI: The "Trigger" Button ---
if st.button("Search"):
    if not api_key:
        st.error("Please enter your API Key in the sidebar.")
    elif not user_query:
        st.warning("Please type a question into the search bar first!")
    else:
        os.environ["GROQ_API_KEY"] = api_key
        
        with st.spinner(f"Asking LLM: '{user_query}'..."):
            
            # 1. Retrieve Top Context
            query_vector = embed_model.encode([user_query])
            distances, indices = index.search(np.array(query_vector), k=3)
            retrieved_context = "\n\n".join([speech_chunks[i] for i in indices[0]])

            # 2. Call Groq LLM
            try:
                client = Groq()
                
                # UPDATED PROMPT: Fixed the Dovish definition per Claude's feedback
                system_prompt = """
                You are a strict financial extraction algorithm. 
                CRITICAL RULES:
                1. Base answer EXCLUSIVELY on text inside <document> tags.
                2. 'Hawkish': Supporting rate hikes. 'Dovish': Supporting rate cuts or warning against the dangers of raising rates. 'Neutral': No change.
                Respond ONLY with JSON: {"macro_signal": "Hawkish/Dovish/Neutral", "key_driver": "Direct quote from text", "confidence_score": "Score based on this strict rubric: 100 = Quote explicitly and directly answers the question. 75 = Quote is highly relevant but requires slight inference. 50 = Quote is tangentially related but weak. 10 = Answer not found in text."}
                """
                
                chat_completion = client.chat.completions.create(
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": f"<document>\n{retrieved_context}\n</document>\n\nAnalyze document for this query: {user_query}"}
                    ],
                    model="llama-3.3-70b-versatile",
                    temperature=0.1, 
                    response_format={"type": "json_object"}
                )
                
                final_output = json.loads(chat_completion.choices[0].message.content)
                
                # 3. Python Safety Net (Regex)
                extracted_quote = final_output.get("key_driver", "")
                def strip_punctuation(text):
                    return " ".join(re.sub(r'[^\w\s]', '', text.lower()).split())
                
                clean_quote = strip_punctuation(extracted_quote)
                clean_context = strip_punctuation(retrieved_context)

                safety_net_passed = False 

                if clean_quote in clean_context:
                    safety_net_passed = True
                else:
                    quote_words = set(clean_quote.split())
                    context_words = set(clean_context.split())
                    if len(quote_words) > 0:
                        overlap = quote_words.intersection(context_words)
                        if len(overlap) / len(quote_words) >= 0.90:
                            safety_net_passed = True

                # --- UI: Display Results ---
                st.subheader("📊 Final Output")
                st.json(final_output)
                
                # The Expandable Evidence Section
                with st.expander("🔍 View Source Evidence"):
                    st.markdown(f"**Source URL:** [Live Fed Speech]({latest_speech_url})")
                    st.markdown("**Retrieved Context Chunks:**")
                    st.info(retrieved_context)
                                
                if safety_net_passed:
                    st.success("✅ PASSED: Quote verified mathematically against retrieved chunks.")
                else:
                    st.error("❌ FAILED: Hallucination Detected! The LLM used outside knowledge.")

            except Exception as e:
                st.error(f"LLM Call Failed: {e}")