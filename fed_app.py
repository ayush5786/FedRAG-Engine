import streamlit as st
from ingestion import fetch_valid_speeches
from rag_engine import semantic_chunking, build_faiss_index, retrieve_top_k, hallucination_check, generate_macro_signal

st.set_page_config(page_title="FedRAG Engine", page_icon="🏦", layout="centered")

st.title("🏦 FedRAG Engine")
st.subheader("Deterministic Macroeconomic Signal Extraction")

# Handle Secrets securely with a Sidebar fallback
api_key = st.secrets.get("GROQ_API_KEY")

with st.sidebar:
    st.header("⚙️ Configuration")
    if not api_key:
        st.info("No system API key found. Please provide your own to test the app.")
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        if not api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()
    else:
        st.success("API key securely loaded from server environment.")

# --- CACHE 1: Data Ingestion (TTL = 1 hour) ---
@st.cache_data(ttl=3600)
def get_clean_data():
    rss_url = "https://www.federalreserve.gov/feeds/speeches.xml"
    return fetch_valid_speeches(rss_url, max_speeches_to_check=5)

# --- CACHE 2: Vector DB Build (TTL = 1 hour) ---
@st.cache_resource(ttl=3600)
def init_vector_db(_speeches):
    all_chunks = []
    for s in _speeches:
        all_chunks.extend(semantic_chunking(s['text']))
    
    if not all_chunks:
        return None, None
        
    return build_faiss_index(all_chunks)

# Execute Data & DB Load
with st.spinner("Syncing live data from Federal Reserve..."):
    speeches = get_clean_data()
    index, chunks = init_vector_db(speeches)

if not index:
    st.warning("No macroeconomic data passed the ingestion filters today.")
    st.stop()

# --- UI INTERACTION ---
query = st.text_input("Ask a macro question (e.g., 'What is the stance on inflation?'):")

if st.button("Extract Signal") and query:
    with st.spinner("Retrieving semantic chunks & generating signal..."):
        # 1. Retrieve
        top_chunks = retrieve_top_k(query, index, chunks, k=3)
        combined_context = " ".join(top_chunks)
        
        # 2. Generate
        result = generate_macro_signal(query, top_chunks, api_key)
        
        if "error" in result:
            st.error(result["error"])
        else:
            # 3. Validate Hallucination via Set Theory
            overlap = hallucination_check(result.get("exact_quote", ""), combined_context)
            
            if overlap < 0.90:
                st.error(f"🚨 Hallucination Blocked! The LLM generated a quote with only {overlap*100:.1f}% mathematical overlap with the source data.")
            else:
                # 4. Render output
                st.success("Signal Extracted Successfully")
                st.json(result)
                
                # Traceability Expander
                with st.expander("🔍 View Source Evidence (Top-3 Chunks)"):
                    st.write("The AI generated this JSON using exclusively the following vectors:")
                    for i, chunk in enumerate(top_chunks):
                        st.info(f"**Chunk {i+1}:** {chunk}")
