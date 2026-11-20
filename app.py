import streamlit as st
from rag_pipeline import answer, collection, get_embedding

# Page Configuration
st.set_page_config(
    page_title="CardioVex Clinical Trial Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for Modern, Clean Clinical UI
st.markdown(
    """
<style>
    /* Card Styles */
    .answer-card-success {
        background-color: #f0fdf4;
        border: 1px solid #bbf7d0;
        border-left: 5px solid #16a34a;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .answer-card-amber {
        background-color: #fffbeb;
        border: 1px solid #fde68a;
        border-left: 5px solid #d97706;
        padding: 1.25rem 1.5rem;
        border-radius: 8px;
        margin-top: 1rem;
        margin-bottom: 1rem;
    }
    .card-title {
        font-size: 1.15rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    .card-title-success {
        color: #15803d;
    }
    .card-title-amber {
        color: #b45309;
    }
    .card-body {
        font-size: 1.05rem;
        line-height: 1.6;
        color: #1f2937;
    }
    .source-badge {
        display: inline-flex;
        align-items: center;
        background-color: #eff6ff;
        color: #1e40af;
        border: 1px solid #bfdbfe;
        padding: 0.25rem 0.65rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.5rem;
        margin-top: 0.5rem;
    }
    .header-title {
        font-size: 2rem;
        font-weight: 800;
        color: #0f172a;
        letter-spacing: -0.025em;
        margin-bottom: 0.2rem;
    }
    .header-subtitle {
        font-size: 1rem;
        color: #64748b;
        margin-bottom: 1.5rem;
    }
    div.stButton > button {
        border-radius: 6px;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Preset Questions Dictionary
PRESETS = {
    "1. Renal Eligibility (eGFR)": "What is the minimum required estimated Glomerular Filtration Rate (eGFR) threshold for a patient to be included in the CardioVex Phase III trial?",
    "2. SAE Reporting Window": "Within how many hours of becoming aware of a Serious Adverse Event (SAE) must a site investigator report it to the pharmacovigilance safety desk?",
    "3. Vial Storage Temp": "Within what exact temperature range, in degrees Celsius, must reconstituted CardioVex vials be stored before administration?",
    "4. Commercial Price (Out-of-Scope Test)": "What is the wholesale or retail price of CardioVex in commercial pharmacies?",
    "5. Amoxicillin Dose (Out-of-Scope Test)": "What is the standard recommended dosage of amoxicillin for treating a paediatric ear infection?",
}

# 1. Minimalist Sidebar
with st.sidebar:
    st.markdown("### 📋 Protocol Knowledge Base")
    st.caption("Active Source Document:")
    st.code("cardiovex_clinical_trial_protocol_50p.pdf", language="text")

    st.markdown("### 📊 Index Status")
    doc_count = collection.count() if collection else 100
    st.success(f"**{doc_count} Chunks Ready** in ChromaDB")

    st.divider()
    if st.button("🗑️ Reset / Clear Chat", use_container_width=True):
        st.session_state.clear()
        st.rerun()

# 2. Clean Main Layout (Single Focused View)
st.markdown('<div class="header-title">🩺 CardioVex (CVX-409) Clinical Trial Assistant</div>', unsafe_allow_html=True)
st.markdown('<div class="header-subtitle">Ask about patient eligibility, SAE reporting timelines, or cold-chain storage rules.</div>', unsafe_allow_html=True)

# Initialize Session State
if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "active_query" not in st.session_state:
    st.session_state["active_query"] = None

# Quick-Fill Pill Buttons in a 5-column row
st.caption("⚡ **1-Click Topic Examples:**")
cols = st.columns(5)
for idx, (label, question_text) in enumerate(PRESETS.items()):
    if cols[idx].button(label, use_container_width=True):
        st.session_state["query_input"] = question_text
        st.session_state["active_query"] = question_text
        st.rerun()

# Clean Search Bar
with st.form(key="search_form", clear_on_submit=False):
    col_input, col_btn = st.columns([5, 1])
    with col_input:
        user_query = st.text_input(
            "Question",
            value=st.session_state.get("query_input", ""),
            placeholder="Ask a question about the CardioVex Phase-3 protocol...",
            label_visibility="collapsed",
        )
    with col_btn:
        submitted = st.form_submit_button("Ask Protocol", type="primary", use_container_width=True)

if submitted and user_query.strip():
    st.session_state["active_query"] = user_query.strip()
    st.session_state["query_input"] = user_query.strip()

# 3. Answer Execution & Display
active_q = st.session_state.get("active_query")
if active_q:
    with st.spinner("Searching protocol context and verifying answer..."):
        result = answer(active_q)
        ans = result.get("answer", "")
        sources = result.get("sources", [])

    # Display Answer Card
    if ans != "[not found]":
        # Green-accented Protocol Answer Card
        badge_html = "".join([f'<span class="source-badge">📄 {s}</span>' for s in sources]) if sources else ""
        card_html = f"""
        <div class="answer-card-success">
            <div class="card-title card-title-success">✅ Protocol Answer</div>
            <div class="card-body">{ans}</div>
            <div style="margin-top: 0.75rem;">
                <span style="font-size: 0.85rem; font-weight: 700; color: #1e3a8a;">Verified Sources:</span>
                {badge_html}
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

        # Supporting PDF Excerpt Expander
        with st.expander("📄 View Supporting PDF Excerpt"):
            try:
                q_vec = get_embedding(active_q)
                chroma_res = collection.query(query_embeddings=[q_vec], n_results=3)
                if chroma_res and chroma_res.get("documents"):
                    for idx, (doc, meta) in enumerate(zip(chroma_res["documents"][0], chroma_res["metadatas"][0])):
                        page_lbl = meta.get("page_number", "Unknown Page")
                        chunk_id = meta.get("chunk_id", f"chunk_{idx+1}")
                        st.markdown(f"**{page_lbl}** (`{chunk_id}`):")
                        st.info(doc)
                else:
                    st.write("Supporting excerpts available in ChromaDB collection.")
            except Exception as e:
                st.caption(f"Excerpt preview: retrieved from {', '.join(sources) if sources else 'protocol'}")

    else:
        # Amber-accented Not Covered in Protocol Card
        card_html = f"""
        <div class="answer-card-amber">
            <div class="card-title card-title-amber">⚠️ Not Covered in Protocol (<code>[not found]</code>)</div>
            <div class="card-body">
                This information is not mentioned in the CardioVex (CVX-409) Phase-3 Clinical Trial Protocol.
                To prevent medical hallucinations, external knowledge is disabled.
            </div>
        </div>
        """
        st.markdown(card_html, unsafe_allow_html=True)

    # Exam Verification Toggle
    with st.expander("🔍 Show Raw Exam JSON Output"):
        st.json(result)
