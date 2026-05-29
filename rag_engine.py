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

def auto_route_speeches(query, speech_pool, api_key, top_n=3):
    """
    Analyzes speech headlines against the user's query using a fast LLM call
    to automatically select the most relevant documents.
    """
    from groq import Groq
    import json
    
    client = Groq(api_key=api_key)
    
    # Format the pool into a clean directory for the LLM to read
    directory = [{"id": i, "title": s["title"], "date": s["date"]} for i, s in enumerate(speech_pool)]
    
    system_prompt = f"""
    You are an advanced document routing agent. Your job is to analyze a user's macroeconomic query and select the top {top_n} most relevant Federal Reserve speeches from the provided directory based ONLY on their titles and dates.
    
    Return a strictly formatted JSON object containing an array of the selected speech IDs:
    {{
        "selected_ids": [0, 2, 5]
    }}
    If absolutely no speeches match the topic, return an empty array. Do not return markdown blocks or prose.
    """
    
    user_prompt = f"User Query: {query}\n\nSpeech Directory:\n{json.dumps(directory)}"
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            response_format={"type": "json_object"}
        )
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("selected_ids", [])
    except Exception as e:
        print(f"Routing error: {e}")
        return []

def generate_general_semantic_signal(query, context_chunks, api_key):
    """
    Dynamically analyzes the text cohort against the user's custom query,
    forcing a native JSON response format from Groq.
    """
    client = Groq(api_key=api_key)
    combined_context = "\n\n".join(context_chunks)
    
    system_prompt = """
    You are an elite central bank research analyst. Your job is to extract explicit thematic signals from Federal Reserve transcripts based strictly on the user's query.

    You must respond with a strictly formatted JSON object matching this schema exactly. Do not output any conversational prose or markdown formatting outside of the JSON object.
    {
        "theme": "The core topic identified from the user's query (e.g., AI Expansion, Trade Relations, Inflation, etc.)",
        "stance": "The Fed speaker's precise stance/sentiment on this theme. Must be exactly one of: [Positive/Accommodative, Neutral/Balanced, Negative/Restrictive, Out of Scope]",
        "rationale": "A concise, 2-3 sentence economic analysis explaining WHY the speaker holds this stance, based ONLY on the text.",
        "exact_quote": "The exact, verbatim sentence from the text that proves this stance. If the text genuinely does not discuss the user's topic, output 'N/A'."
    }

    CRITICAL RULES:
    1. Evaluate the stance strictly on the theme requested.
    2. Do not hallucinate. If the text lacks explicit information to answer the query, set stance to 'Out of Scope' and exact_quote to 'N/A'.
    """
    
    user_prompt = f"""
    User Query: {query}
    
    Provided Speech Context:
    {combined_context}
    """
    
    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
            # This completely solves formatting bugs by forcing a structured JSON output
            response_format={"type": "json_object"} 
        )
        
        raw_content = response.choices[0].message.content.strip()
        return json.loads(raw_content)
        
    except Exception as e:
        return {"error": f"Failed to parse signal: {str(e)}"}
