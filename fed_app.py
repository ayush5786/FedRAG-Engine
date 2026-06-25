import streamlit as st
import re
from groq import Groq
from ingestion import fetch_speech_list, scrape_speech_text
from rag_engine import (
    semantic_chunking, build_faiss_index, retrieve_top_k, 
    hallucination_check, generate_general_semantic_signal
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
            "Qwen 27B (Balanced, High Limits)"
        ],
        help="If you receive a 'Rate Limit' error, switch to the Qwen 27B model for an isolated token pool and high capacity."
    )
    
    # Map the UI choice to the actual Groq model ID
    if "70B" in model_choice:
        selected_model = "llama-3.3-70b-versatile"
    else:
        selected_model = "qwen/qwen3.6-27b"

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
st.markdown("### 💬 Query the Engine")

# Expandable hint box for new users
with st.expander("💡 Not sure what to ask? Try one of these themes:"):
    st.markdown("""
    * **Inflation & Rates:** *"What is the current stance on sticky services inflation and rate cuts?"*
    * **Geopolitics:** *"How are global trade tensions and tariffs impacting domestic supply chains?"*
    * **Technology & Labor:** *"What are Fed governors saying about AI, automation, and labor productivity?"*
    * **Housing:** *"What is the outlook on the commercial real estate sector and housing affordability?"*
    """)

# Use columns to put the text box and button side-by-side
col1, _ = st.columns([6, 1])    
with col1:
    query = st.chat_input("Ask about inflation, AI, geopolitics, or labor...")

# Core Execution Block
# The `query.strip()` ensures we don't accidentally run empty searches
if (query) and query.strip():
    
    # STEP A: Vectorize EVERYTHING in your current slider horizon right away
    with st.spinner("Scraping and building full vector index pool..."):
        all_urls = tuple([meta['url'] for meta in speech_pool])
        index, chunks = lazy_load_and_index_cohort(all_urls)
        
    if isinstance(index, str) or not index:
        st.error("Could not process text data for the speech directory.")
        st.stop()
        
    # STEP B: Let FAISS search the actual text transcripts mathematically
    with st.spinner("Sifting through speech transcripts for semantic matches..."):
        top_chunks = retrieve_top_k(query, index, chunks, k=3)
        combined_context = " ".join(top_chunks)
        
    # Inform the user that the FAISS Pipeline successfully extracted matches
    st.write(query)
    st.info(f"🤖 **FAISS Pipeline Active:** Successfully isolated the top 3 contextual match-blocks from full transcripts.")

    # STEP C: Directly extract insights using our top chunks context
    with st.spinner("Extracting semantic insights..."):
        result = generate_general_semantic_signal(query, top_chunks, api_key, model_name=selected_model)
        
    # ====================================================================
    # 🎨 STEP D: Render Output Data & Handle Out-of-Scope Pivots
    # ====================================================================
    if "error" in result:
        error_msg = result["error"]
        
        # Check if it's a standard Groq Rate Limit (429)
        if "429" in error_msg or "rate_limit_exceeded" in error_msg:
            wait_time_match = re.search(r"try again in ([0-9a-zA-Z.]+)", error_msg)
            if wait_time_match:
                raw_time = wait_time_match.group(1)
                clean_time = re.sub(r'\.\d+', '', raw_time)
                wait_info = f"**{clean_time}**"
            else:
                wait_info = "a few minutes"
            
            st.markdown(
                "<h3 style='text-align: center; color: #ff4b4b;'>🛑 70B Model Limit Reached</h3>", 
                unsafe_allow_html=True
            )
            st.markdown(f"""
            You have exhausted your daily free-tier token allocation for the heavy **Llama 3.3 70B** model. 
            
            ⏳ **Rolling Limit Warning:** You will regain enough capacity to run *one more query* in {wait_info}. However, your full daily allowance will not completely reset until midnight UTC.
            
            ---
            
            #### 💡 How to keep testing right now:
            Switch the **AI Engine Selection** dropdown on the left sidebar to **Llama 4 Scout 17B (Balanced, High Limits)**. 
            It has a much higher daily token allowance and will process your query instantly!
            """)
        else:
            st.error("### 🚨 Extraction Pipeline Snag")
            st.code(error_msg, language="bash")
            
    else:
        theme = result.get("theme", "N/A")
        stance = result.get("stance", "Out of Scope")
        exact_quote = result.get("exact_quote", "")
        
        # -------------------------------------------------------------
        # THE MACRO PIVOT: Triggered when the AI determines the topic is missing
        # -------------------------------------------------------------
        if stance == "Out of Scope" or exact_quote.upper() == "N/A":
            st.warning(f"🔍 The Federal Reserve has not explicitly discussed the specific theme of your question (**'{query}'**) in these recent speeches.")
            st.divider()
            
            with st.spinner("Compiling current central bank priority briefing instead..."):
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
                    Analyze the following recent speech transcript chunks and generate a crisp, highly professional 3-bullet-point Executive Briefing summarizing what the speakers are ACTUALLY prioritizing right now.
                    Text Context: {combined_fallback_text}
                    """
                    try:
                        brief_response = client.chat.completions.create(
                            model=selected_model,
                            messages=[{"role": "user", "content": briefing_prompt}],
                            temperature=0.2
                        )
                        st.subheader("📋 Current Fed Priority Briefing")
                        st.success("While your exact theme was out of scope, here is what central bankers are actively focusing on:")
                        st.markdown(brief_response.choices[0].message.content)
                    except Exception as e:
                        st.error(f"Could not synthesize priority summary: {e}")
                else:
                    st.error("Fallback scraping returned no readable document text chunks to analyze.")
                    
        # -------------------------------------------------------------
        # STANDARD EXTRACTION: Render the valid macroeconomic data
        # -------------------------------------------------------------
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
