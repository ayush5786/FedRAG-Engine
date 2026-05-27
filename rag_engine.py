import os
import json
import faiss
import numpy as np
import string
from sentence_transformers import SentenceTransformer
from groq import Groq

# Initialize embedding model globally so it's ready for the app
embedder = SentenceTransformer('all-MiniLM-L6-v2')

def semantic_chunking(text, max_words=300):
    """Splits text by double newlines to preserve paragraph integrity."""
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for p in paragraphs:
        words = p.split()
        if current_word_count + len(words) > max_words:
            chunks.append(" ".join(current_chunk))
            current_chunk = [p]
            current_word_count = len(words)
        else:
            current_chunk.append(p)
            current_word_count += len(words)
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

def build_faiss_index(chunks):
    """Embeds chunks and creates an in-memory L2 FAISS index."""
    embeddings = embedder.encode(chunks)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings).astype('float32'))
    return index, chunks

def retrieve_top_k(query, index, chunks, k=3):
    """Semantic search to find the Top-3 chunks."""
    query_vector = embedder.encode([query]).astype('float32')
    distances, indices = index.search(query_vector, k)
    retrieved = [chunks[i] for i in indices[0]]
    return retrieved

def hallucination_check(llm_quote, retrieved_text):
    """Set-Theory validation: Requires >90% token overlap between quote and source."""
    def clean_and_set(text):
        text = text.translate(str.maketrans('', '', string.punctuation)).lower()
        return set(text.split())
        
    quote_set = clean_and_set(llm_quote)
    source_set = clean_and_set(retrieved_text)
    
    if not quote_set:
        return 0.0
        
    intersection = quote_set.intersection(source_set)
    overlap_percentage = len(intersection) / len(quote_set)
    return overlap_percentage

def generate_macro_signal(query, retrieved_chunks, api_key):
    """Passes context to Llama-3.3-70B using a strict JSON format and Macro Rubric."""
    client = Groq(api_key=api_key)
    context = "\n\n---\n\n".join(retrieved_chunks)


    system_prompt = """
    You are an expert Quantitative Analyst. Analyze the Federal Reserve text and answer the user's query.
    You must output ONLY valid JSON.
    
    MACROECONOMIC RUBRIC:
    - Strongly Hawkish: Explicit plans to raise rates or strict inflation warnings.
    - Moderately Hawkish: Leaning toward restrictive policy, but data-dependent.
    - Neutral: Balanced risks, maintaining current levels.
    - Moderately Dovish: Leaning toward rate cuts, acknowledging economic slowing.
    - Strongly Dovish: Explicit plans to cut rates or inject liquidity.
    - Out of Scope: If the provided context does not explicitly contain enough information to answer the user's historical or unrelated query.
    
    JSON SCHEMA:
    {
        "signal": "Strongly Hawkish | Moderately Hawkish | Neutral | Moderately Dovish | Strongly Dovish | Out of Scope",
        "rationale": "A 2-sentence explanation of the stance.",
        "exact_quote": "A specific quote extracted verbatim from the text that proves your signal. If the signal is Out of Scope, set this exactly to 'N/A'."
    }
    """
    
    
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Context:\n{context}\n\nQuery: {query}"}
        ],
        temperature=0.1,
        response_format={"type": "json_object"}
    )
    
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"error": "Failed to generate valid JSON"}
