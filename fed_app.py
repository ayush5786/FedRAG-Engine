import streamlit as st
import re
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
    num_speeches = st.slider("Recent Speeches to Scan", min_value=5, max_value=50, value=15)

    st.sidebar.divider()
    st.sidebar.markdown("### 🧠 AI Engine Selection")
    model_choice = st.sidebar.selectbox(
        "Choose Extraction Model:",
        options=[
            "Llama 3.3 70B (Deep Reasoning, Strict Limits)", 
            "Llama 4 Scout 17B (Balanced, High Limits)"
        ],
        help="If you receive a 'Rate Limit' error, switch to the 17B model for higher daily token allowances."
    )
    
    # Map the UI choice to the actual Groq model ID
    if "70B" in model_choice:
        selected_model = "llama-3.3-70b-versatile"
    else:
        selected_model = "meta-llama/llama-4-scout-17b-16e-instruct"

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
st.markdown("Ask any thematic macro question (e.g., 'What is the outlook on AI productivity?'):")
# Use columns to put the text box and button side-by-side
col1, col2 = st.columns([6, 1])
with col1:
    query = st.text_input("query", label_visibility="collapsed")
with col2:
    submit_btn = st.button("Enter ↵", use_container_width=True, type="primary")

# Core Execution Block
if submit_btn or query:
    # STEP A: Auto-Route and Select Speeches matching the query
    with st.spinner("Analyzing directory to isolate relevant speeches..."):
        selected_ids = auto_route_speeches(query, speech_pool, api_key, top_n=3)
        if selected_ids is None:
            st.error("🚨 **Groq API Connection / Rate Limit Issue:** The system could not complete the routing request. Please wait a moment and try refreshing your query.")
            st.stop() 
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
            
            col1, col2 = st.columns(2)
            
            with col1:
                with st.container(border=True):
                    st.markdown("#### 👈 Option 1: Expand Horizon")
                    st.write("Increase **'Recent Speeches to Scan'** on the sidebar to sift deeper into historical central bank remarks.")
                    st.write("") 
                    
            with col2:
                with st.container(border=True):
                    st.markdown("#### ✨ Option 2: Macro Pivot")
                    st.write("Stay on this current timeline window and pivot to extract what the Fed is actually prioritizing instead.")
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
                            model=selected_model,
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
                    
        st.stop() 
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
        result = generate_general_semantic_signal(query, top_chunks, api_key, model_name=selected_model)
        
    # ====================================================================
    # 🎨 STEP D: Render Output Data (With Polished Error Handling)
    # ====================================================================
    if "error" in result:
        error_msg = result["error"]
        
        # Check if it's a standard Groq Rate Limit (429)
        # Check if it's a standard Groq Rate Limit (429)
        if "429" in error_msg or "rate_limit_exceeded" in error_msg:
            
            # Extract the wait time and strip out the ugly decimals
            wait_time_match = re.search(r"try again in ([0-9a-zA-Z.]+)", error_msg)
            if wait_time_match:
                raw_time = wait_time_match.group(1) # e.g., "10m46.271999999s"
                clean_time = re.sub(r'\.\d+', '', raw_time) # Removes .271999999
                wait_info = f"**{clean_time}**"
            else:
                wait_info = "a few minutes"
            
            # 1. The Title (Centered and Streamlit Red)
            st.markdown(
                "<h3 style='text-align: center; color: #ff4b4b;'>🛑 70B Model Limit Reached</h3>", 
                unsafe_allow_html=True
            )
            
            # 2. The Standard Text Underneath (Not Red)
            st.markdown(f"""
            You have exhausted your daily free-tier token allocation for the heavy **Llama 3.3 70B** model. 
            
            ⏳ **Rolling Limit Warning:** You will regain enough capacity to run *one more query* in {wait_info}. However, your full daily allowance will not completely reset until midnight UTC.
            
            ---
            
            #### 💡 How to keep testing right now:
            Switch the **AI Engine Selection** dropdown on the left sidebar to **Llama 4 Scout 17B (Balanced, High Limits)**. 
            It has a much higher daily token allowance and will process your query instantly!
            """)


        else:
            # Fallback for any other unexpected API errors
            st.error("### 🚨 Extraction Pipeline Snag")
            st.code(error_msg, language="bash")
            
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
                
                col1, col2 = st.columns(2)
                with col1:
                    st.markdown("##### 🏷️ Extracted Theme")
                    st.markdown(f"### **{theme}**")
                with col2:
                    st.markdown("##### ⚖️ Calculated Stance")
                    
                    if "Negative" in stance or "Restrictive" in stance:
                        stance_color = "🔴"
                    elif "Positive" in stance or "Accommodative" in stance:
                        stance_color = "🟢"
                    else:
                        stance_color = "⚪" 
                        
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
