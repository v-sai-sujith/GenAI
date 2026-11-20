import json
from pathlib import Path
import streamlit as st
from rag_pipeline import answer, collection

# Set Streamlit Page Configuration
st.set_page_config(
    page_title="CardioVex Protocol & Pharmacovigilance Assistant",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
<style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .status-badge {
        display: inline-block;
        padding: 0.25rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.85rem;
        font-weight: 600;
    }
    .status-found {
        background-color: #DEF7EC;
        color: #03543F;
    }
    .status-not-found {
        background-color: #FDE8E8;
        color: #9B1C1C;
    }
    .source-tag {
        display: inline-block;
        background-color: #E0E7FF;
        color: #3730A3;
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-weight: 600;
        font-size: 0.85rem;
        margin-right: 0.4rem;
        margin-top: 0.3rem;
    }
</style>
""",
    unsafe_allow_html=True,
)

# Sidebar
with st.sidebar:
    st.image(
        "https://img.icons8.com/fluency/96/caduceus.png",
        width=64,
    )
    st.title("System Specs")
    st.markdown("**Project 6:** Clinical Trial Protocol & Pharmacovigilance Assistant")
    st.divider()

    st.subheader("📚 Knowledge Base")
    st.caption("Sole Grounding Source:")
    st.code("cardiovex_clinical_trial_protocol_50p.pdf", language="text")

    doc_count = collection.count() if collection else 0
    st.metric(label="ChromaDB Indexed Chunks", value=doc_count)

    st.subheader("🛡️ Grounding Safeguards")
    st.markdown(
        "- **Zero Hallucination:** Answers strictly restricted to retrieved context.\n"
        "- **Exact Not-Found:** Returns `[not found]` for out-of-domain or ungrounded questions.\n"
        "- **Top-k Chunks:** 5 retrieved via `text-embedding-004`.\n"
        "- **Inference:** `gemini-2.5-flash` (temperature=0.0)."
    )

    st.divider()
    st.caption("CardioVex Phase III Clinical Protocol Assistant")

# Main Title
st.markdown(
    '<div class="main-header">🩺 CardioVex Clinical Trial Protocol Assistant</div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="sub-header">Automated protocol compliance and pharmacovigilance safety query engine grounded strictly in the 50-page Phase III protocol.</div>',
    unsafe_allow_html=True,
)

tab1, tab2 = st.tabs(["💬 Interactive Protocol Query", "📊 Assessment Benchmarks"])

PRESET_QUESTIONS = [
    "What is the minimum required estimated Glomerular Filtration Rate (eGFR) threshold for a patient to be included in the CardioVex Phase III trial?",
    "Within how many hours of becoming aware of a Serious Adverse Event (SAE) must a site investigator report it to the pharmacovigilance safety desk?",
    "What is the wholesale or retail price of CardioVex in commercial pharmacies?",
    "Within what exact temperature range, in degrees Celsius, must reconstituted CardioVex vials be stored before administration?",
    "What is the standard recommended dosage of amoxicillin for treating a paediatric ear infection?",
]

with tab1:
    st.subheader("Ask a Question")
    st.caption(
        "Select a preset question from the benchmark or enter your own clinical protocol question below."
    )

    selected_preset = st.selectbox(
        "⚡ Quick Select Preset Benchmark Question:",
        ["-- Select a question or type below --"] + PRESET_QUESTIONS,
    )

    default_query = ""
    if selected_preset != "-- Select a question or type below --":
        default_query = selected_preset

    query = st.text_area(
        "Enter Clinical or Pharmacovigilance Question:",
        value=default_query,
        height=90,
        placeholder="e.g., What are the exclusion criteria related to liver function tests?",
    )

    col1, col2 = st.columns([1, 5])
    with col1:
        submit = st.button("🔍 Submit Query", type="primary", use_container_width=True)

    if submit and query.strip():
        with st.spinner("Retrieving protocol context and generating grounded response..."):
            result = answer(query.strip())
            ans = result.get("answer", "")
            sources = result.get("sources", [])

        st.markdown("### Response")
        if ans == "[not found]":
            st.error("⚠️ **[not found]**")
            st.info(
                "This information is not present or mentioned in the CardioVex clinical trial protocol. "
                "Per safety guardrails, no speculative answer was generated."
            )
        else:
            st.success("✅ **Grounded Protocol Answer:**")
            st.write(ans)

            st.markdown("#### 📖 Cited Source Pages:")
            if sources:
                src_html = "".join([f'<span class="source-tag">📄 {s}</span>' for s in sources])
                st.markdown(src_html, unsafe_allow_html=True)
            else:
                st.write("No specific source pages.")

with tab2:
    st.subheader("📋 Assessment Benchmark Results (Outputs)")
    st.caption("Verification results across all 5 benchmark questions.")

    outputs_file = Path("outputs.json")
    if outputs_file.exists():
        with open(outputs_file, "r", encoding="utf-8") as f:
            benchmarks = json.load(f)

        for item in benchmarks:
            q_id = item.get("id")
            q_text = item.get("question")
            ans = item.get("answer")
            src = item.get("sources", [])
            exp_ans = item.get("expected_answer")
            exp_src = item.get("expected_sources", [])

            is_match = (ans == "[not found]" and exp_ans == "[not found]") or (src == exp_src)

            with st.expander(f"{q_id}: {q_text[:75]}...", expanded=True):
                st.markdown(f"**Question:** {q_text}")
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("**Expected:**")
                    st.write(f"- Answer: `{exp_ans}`")
                    st.write(f"- Sources: `{exp_src}`")
                with c2:
                    st.markdown("**Actual Generated:**")
                    st.write(f"- Answer: `{ans}`")
                    st.write(f"- Sources: `{src}`")
                if is_match:
                    st.success("Status: ✅ MATCHED")
                else:
                    st.warning("Status: REVIEW")
    else:
        st.warning("outputs.json not found. Run `python3 main.py` first.")
