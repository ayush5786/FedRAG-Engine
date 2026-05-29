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
    Analyzes speech headlines AND RSS content summaries against the user's query 
    using a fast LLM call to automatically select the most relevant documents.
    """
    
    client = Groq(api_key=api_key)
    
    # Format the pool into a clean directory, now passing the rich 'summary' context
    directory = [
        {
            "id": i, 
            "title": s["title"], 
            "date": s["date"],
            "summary_context": s.get("summary", "No summary available.")
        } 
        for i, s in enumerate(speech_pool)
    ]
    
    system_prompt = f"""
    You are an advanced document routing agent. Your job is to analyze a user's macroeconomic query and select the top {top_n} most relevant Federal Reserve speeches from the provided directory.
    
    Use the speech title, date, AND the 'summary_context' block to make your determination. The 'summary_context' explains the actual themes of the speech, allowing you to route accurately even if the title is generic or ambiguous.
    
    Return a strictly formatted JSON object containing an array of the selected speech IDs:
    {{
        "selected_ids": [0, 2, 5]
    }}
    If absolutely no speeches match the topic, return an empty array. Do not return markdown blocks or prose. Always frame your answer by explicitly stating 'According to recent Federal Reserve remarks...'
    """
    
    user_prompt = f"User Query: {query}\n\nSpeech Directory:\n{json.dumps(directory)}"
    
    try:
        response = client.chat.completions.create(
            model="meta-llama/llama-4-scout-17b-16e-instruct",
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
        return None

def generate_general_semantic_signal(query, retrieved_chunks, api_key, model_name="llama-3.3-70b-versatile"):
    """
    Executes the heavy macroeconomic text extraction. 
    Accepts a dynamic model_name from the UI to safely handle free-tier limit fallbacks.
    """
    from groq import Groq
    import json
    
    client = Groq(api_key=api_key)
    
    # Combine the top text chunks into a unified context payload
    context_block = "\n\n".join([f"[Source Chunk]: {chunk}" for chunk in retrieved_chunks])
    
    system_prompt = """
    You are an expert macroeconomic analyst. Your task is to evaluate the provided Federal Reserve speech contexts against the user's query.
    
    You must extract:
    1. The core theme discussed.
    2. The calculated monetary policy stance. Strict allowed values: 'Positive/Accommodative' or 'Negative/Restrictive'.
    3. An analytical rationale formatted as a single, well-written, and professional paragraph (do NOT use bullet points).
    4. An exact, full-length verbatim quote (at most 2 to 3 full sentences) directly from the text that deeply supports your analysis. Do not truncate the quote.
    
    Return a strictly formatted JSON object:
    {
        "theme": "Inflation",
        "stance": "Negative/Restrictive",
        "rationale": "A single narrative paragraph breaking down the central bank speaker's analytical stance and underlying economic concerns without any lists or bullet points.",
        "exact_quote": "A long, multi-sentence verbatim extraction from the source text that provides deep context. It should be at least two full sentences."
    }
    """
    
    user_prompt = f"User Query: {query}\n\nRetrieved Context Documents:\n{context_block}"
    
    try:
        response = client.chat.completions.create(
            model=model_name,  # Dynamically passed from the Streamlit frontend dropdown
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.1,
            response_format={"type": "json_object"}
        )
        return json.loads(response.choices[0].message.content.strip())
    except Exception as e:
        print(f"Extraction error: {e}")
        return {"error": str(e)}
