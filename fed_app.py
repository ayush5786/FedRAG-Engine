import streamlit as st
from ingestion import fetch_valid_speeches
from rag_engine import semantic_chunking, build_faiss_index, retrieve_top_k, hallucination_check, generate_macro_signal

st.set_page_config(page_title="FedRAG Engine", page_icon="🏦", layout="centered")

st.title("🏦 FedRAG Engine")
st.subheader("Deterministic Macroeconomic Signal Extraction")

# --- 1. SIDEBAR: Config & Transparency ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.secrets.get("GROQ_API_KEY")
    if not api_key:
        st.info("No system API key found. Please provide your own.")
        api_key = st.text_input("Enter your Groq API Key:", type="password")
        if not api_key:
            st.warning("Please enter your API key to proceed.")
            st.stop()
    else:
        st.success("API key securely loaded from server environment.")

    st.divider()
    
    st.header("🎛️ Database Settings")
    # THE SLIDER: Lets the user choose how many speeches to analyze
    num_speeches = st.slider("Recent Speeches to Analyze", min_value=1, max_value=10, value=5)

# --- 2. CACHE & DATA LOAD (Dynamic based on Slider) ---
# Notice we pass 'max_speeches' into the function now!
@st.cache_data(ttl=3600)
def get_clean_data(max_speeches):
    rss_url = "https://www.federalreserve.gov/feeds/speeches.xml"
    return fetch_valid_speeches(rss_url, target_valid_speeches=max_speeches)

@st.cache_resource(ttl=3600)
def init_vector_db(_speeches):
    all_chunks = []
    for s in _speeches:
        all_chunks.extend(semantic_chunking(s['text']))
    if not all_chunks:
        return None, None
    return build_faiss_index(all_chunks)

# Execute Data & DB Load using the slider value
with st.spinner(f"Syncing top {num_speeches} speeches from the Fed..."):
    speeches = get_clean_data(num_speeches)
    index, chunks = init_vector_db(speeches)

if not index:
    st.warning("No macroeconomic data passed the ingestion filters today.")
    st.stop()

# --- 3. SIDEBAR: Transparency UI ---
# Now that 'speeches' is loaded, we print them in the sidebar
with st.sidebar:
    st.divider()
    st.header("📚 Active Data Context")
    st.caption("These speeches passed the macroeconomic filter and are powering the AI:")
    for i, speech in enumerate(speeches):
        # We added the date in italics right below the clickable title
        st.markdown(f"**{i+1}.** [{speech['title']}]({speech['url']})  \n*{speech.get('date', 'Unknown Date')}*")

# --- UI INTERACTION ---
query = st.text_input("Ask a macro question (e.g., 'What is the stance on inflation?'):")

# Initialize persistent memory if it doesn't exist yet
if "api_output" not in st.session_state:
    st.session_state.api_output = None
    st.session_state.combined_context = None
    st.session_state.top_chunks = None
    
# Phase 1: The Button (Calculate & Save)
if st.button("Extract Signal") and query:
    with st.spinner("Retrieving semantic chunks & generating signal..."):
        # 1. Retrieve
        top_chunks = retrieve_top_k(query, index, chunks, k=3)
        combined_context = " ".join(top_chunks)
        
        # 2. Generate
        result = generate_macro_signal(query, top_chunks, api_key)
        
        # 3. Save to Persistent Memory (NO UI RENDERING HERE!)
        st.session_state.api_output = result
        st.session_state.combined_context = combined_context
        st.session_state.top_chunks = top_chunks

# Phase 2: The Render (Read & Display)
# Because this is outside the button, it stays visible even if the user changes tabs!
if st.session_state.api_output is not None:
    result = st.session_state.api_output
    combined_context = st.session_state.combined_context
    top_chunks = st.session_state.top_chunks
    
    if "error" in result:
        st.error(result["error"])
    else:
        exact_quote = result.get("exact_quote", "")
        signal = result.get("signal", "")
        
        # --- Bypass the safety net if it's Out of Scope ---
        if signal == "Out of Scope" or exact_quote.upper() == "N/A":
            st.info("ℹ️ Query is out of scope based on recent data. No hallucination check required.")
            st.json(result)
        else:
            # 3. Validate Hallucination via Set Theory
            overlap = hallucination_check(exact_quote, combined_context)
            
            if overlap < 0.90:
                st.error(f"🚨 Hallucination Blocked! The LLM generated a quote with only {overlap*100:.1f}% mathematical overlap with the source data.")
            else:
                # 4. Render output
                st.success("✅ Signal Extracted Successfully")
                st.json(result)
                
                # Traceability Expander
                with st.expander("🔍 View Source Evidence (Top-3 Chunks)"):
                    st.write("The AI generated this JSON using exclusively the following vectors:")
                    for i, chunk in enumerate(top_chunks):
                        st.info(f"**Chunk {i+1}:** {chunk}")
