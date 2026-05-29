import streamlit as st
from groq import Groq
from ingestion import fetch_speech_list, scrape_speech_text
from rag_engine import (
    semantic_chunking, build_faiss_index, retrieve_top_k, 
    hallucination_check, generate_general_semantic_signal, auto_route_speeches
)

st.set_page_config(page_title="FedRAG Engine", page_icon="🏦", layout="centered")

st.title("🏦 FedRAG Engine")
st.subheader("Autonomous Semantic Macro Extraction")

# --- 1. SIDEBAR: Config & Fetch Pool ---
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
    
    st.header("🎛️ Database Pool")
    # Kept default at 15, maxing out at 50
    num_speeches = st.slider("Recent Speeches to Scan", min_value=5, max_value=50, value=15)

# --- 2. FETCH HEADLINES ONLY (Instant) ---
@st.cache_data(ttl=3600)
def get_speech_pool(max_speeches):
    rss_url = "https://www.federalreserve.gov/feeds/speeches.xml"
    return fetch_speech_list(rss_url, target_speeches=max_speeches)

with st.spinner("Syncing latest speech directory..."):
    speech_pool = get_speech_pool(num_speeches)

# --- 3. ON-DEMAND COHORT COUPLING ---
@st.cache_resource(ttl=3600)
def lazy_load_and_index_cohort(urls):
    if not urls:
        return "EMPTY", None
    combined_chunks = []
    for url in urls:
        speech_text, _ = scrape_speech_text(url)
        chunks = semantic_chunking(speech_text)
        if chunks:
            combined_chunks.extend(chunks)
    if not combined_chunks:
        return "EMPTY", None
    return build_faiss_index(combined_chunks)

# --- 4. MAIN INTERACTION UI ---
query = st.text_input("Ask any thematic macro question (e.g., 'What is the outlook on AI productivity?'):")

# Core Execution Block
if query:
    # STEP A: Auto-Route and Select Speeches matching the query
    with st.spinner("Analyzing directory to isolate relevant speeches..."):
        selected_ids = auto_route_speeches(query, speech_pool, api_key, top_n=3)
        selected_metas = [speech_pool[i] for i in selected_ids if i < len(speech_pool)]
        
    # ====================================================================
    # 🔥 LIVE PRODUCTION FALLBACK SYSTEM (DASHBOARD CARD UPGRADE)
    # ====================================================================
    if not selected_metas:
        st.warning(f"🔍 No recent speeches explicitly match the theme of your question: **'{query}'**.")
        st.divider()
        
        trigger_briefing = False
        
        # SCENARIO 1: User has room left to scroll/scan deeper
        if num_speeches < 50:
            st.write("### 💡 Choose what would you like to do next:")
            
            # Modern side-by-side dashboard split
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container(border=True):
                    st.markdown("#### 👈 Option 1: Expand Horizon")
                    st.write("Increase **'Recent Speeches to Scan'** on the sidebar to sift deeper into historical central bank remarks.")
                    # Empty space to visually balance the button in col2
                    st.write("") 
                    
            with col2:
                with st.container(border=True):
                    st.markdown("#### ✨ Option 2: Macro Pivot")
                    st.write("Stay on this current timeline window and pivot to extract what the Fed is actually prioritizing instead.")
                    # Custom styled block width button to look uniform
                    if st.button("Run Executive Briefing", use_container_width=True, type="primary"):
                        trigger_briefing = True
                        
        # SCENARIO 2: User has fully maxed out the scroll horizon to 50
        else:
            st.markdown("⚠️ **Maximum Search Horizon Reached:** You have exhausted the scroll limit (50 speeches). This specific topic has not been addressed by Fed officials in recent months.")
            trigger_briefing = True
        
        # --- THE BRIEFING COMPILER BLOCK ---
        if trigger_briefing:
            with st.spinner("Compiling current central bank priority briefing..."):
                fallback_speeches = speech_pool[:3]
                fallback_urls = [s['url'] for s in fallback_speeches]
                
                combined_fallback_text = ""
                for url in fallback_urls:
                    text, _ = scrape_speech_text(url)
                    if text:
                        combined_fallback_text += f"\n\n--- Document Snippet ---\n{text[:2500]}"
                
                if combined_fallback_text.strip():
                    client = Groq(api_key=api_key)
                    briefing_prompt = f"""
                    The user submitted a query about '{query}', which is completely absent from the current Federal Reserve speech directory pool.
                    
                    Analyze the following recent speech transcript text chunks and generate a crisp, highly professional 3-bullet-point Executive Briefing summarizing the dominant macroeconomic issues and themes the speakers are ACTUALLY prioritizing right now. Ensure it sounds like a research memo.
                    
                    Text Context:
                    {combined_fallback_text}
                    """
                    
                    try:
                        brief_response = client.chat.completions.create(
                            model="llama-3.3-70b-versatile",
                            messages=[{"role": "user", "content": briefing_prompt}],
                            temperature=0.2
                        )
                        st.divider()
                        st.subheader("📋 Current Fed Priority Briefing")
                        st.success("While your exact theme was out of scope, here is what central bankers are actively focusing on:")
                        st.markdown(brief_response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Could not synthesize priority summary: {e}")
                else:
                    st.error("Fallback scraping returned no readable document text chunks to analyze.")
                    
        st.stop() # Soft lock execution here so the downstream RAG stack doesn't trigger
    # ====================================================================
        
    # Display the auto-selected cohort to the user dynamically
    st.info(f"🤖 **Auto-Selected Cohort ({len(selected_metas)} Speeches Match Your Theme):**")
    for meta in selected_metas:
        st.markdown(f"- **{meta['date']}**: [{meta['title']}]({meta['url']})")
        
    # STEP B: Lazy load, Scrape, and Vectorize ONLY the auto-selected speeches
    with st.spinner("Scraping and building isolated vector cohort..."):
        urls = tuple([meta['url'] for meta in selected_metas])
        index, chunks = lazy_load_and_index_cohort(urls)
        
    if isinstance(index, str) or not index:
        st.error("Could not process text data for the auto-selected speeches.")
        st.stop()
        
    # STEP C: RAG Extraction Pipeline
    with st.spinner("Extracting semantic insights..."):
        top_chunks = retrieve_top_k(query, index, chunks, k=3)
        combined_context = " ".join(top_chunks)
        result = generate_general_semantic_signal(query, top_chunks, api_key)
        
    # STEP D: Render Output Data
    if "error" in result:
        st.error(result["error"])
    else:
        theme = result.get("theme", "N/A")
        stance = result.get("stance", "Out of Scope")
        exact_quote = result.get("exact_quote", "")
        
        if stance == "Out of Scope" or exact_quote.upper() == "N/A":
            st.info(f"ℹ️ Analysis concluded: Theme '{theme}' is out of scope within the contextual data.")
            st.json(result)
        else:
            overlap = hallucination_check(exact_quote, combined_context)
            if overlap < 0.90:
                st.error(f"🚨 Hallucination Blocked! Quote mathematical overlap was only {overlap*100:.1f}%.")
            else:
                st.success(f"✅ Target Insights Extracted Successfully")
                
                # --- Polished Text Headers ---
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🏷️ Extracted Theme")
                    st.markdown(f"### **{theme}**")
                with col2:
                    st.markdown("##### ⚖️ Calculated Stance")
                    
                    # Clean, exact matching for your specific terms
                    if "Negative" in stance or "Restrictive" in stance:
                        stance_color = "🔴"
                    elif "Positive" in stance or "Accommodative" in stance:
                        stance_color = "🟢"
                    else:
                        stance_color = "⚪" # Used for Neutral/Balanced or Out of Scope
                        
                    st.markdown(f"### {stance_color} **{stance}**")
                
                st.divider()
                    
                st.markdown("#### 🧠 AI Rationale")
                st.write(result.get("rationale", "No rationale provided."))

                st.write('')
                
                st.markdown("#### 📝 Verbatim Quote")
                quote = result.get('exact_quote', 'No quote found.')
                st.markdown(f"> *\"{quote}\"*")
                
                st.divider()
                
                with st.expander("🔍 View Source Evidence Vectors"):
                    for i, chunk in enumerate(top_chunks):
                        st.info(f"**Chunk {i+1}:** {chunk}")
                with st.expander("⚙️ View Raw JSON Payload (Developer Mode)"):
                    st.json(result)
else:
    st.write("👈 Set your scanning window size in the sidebar and enter a question above to begin analysis.")
