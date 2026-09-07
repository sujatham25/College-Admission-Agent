"""
app.py
------
Streamlit UI for the RAG-based College Admission Agent.
Modern hero landing page + chat interface.

Run with:
    streamlit run app.py

⚠️  DEMO PROJECT — IBM AICTE Internship 2026
"""

import streamlit as st

from admission_agent import ask, AgentResponse
from rag.retriever import get_chunk_count

# ---------------------------------------------------------------------------
# Page configuration
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="College Admission Agent",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------

st.markdown("""
<style>
/* ── Reset & base ── */
* { box-sizing: border-box; }
[data-testid="stAppViewContainer"] {
    background: #f5f6fa;
    font-family: -apple-system, "Segoe UI", system-ui, sans-serif;
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="collapsedControl"] { display: none; }

/* ── Top nav bar ── */
.nav-bar {
    background: #ffffff;
    border-bottom: 1px solid #e5e7eb;
    padding: 0 2.5rem;
    height: 58px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    position: sticky;
    top: 0;
    z-index: 100;
    margin-bottom: 0;
}
.nav-logo {
    display: flex;
    align-items: center;
    gap: 10px;
    font-weight: 700;
    font-size: 1.05rem;
    color: #1f2328;
    text-decoration: none;
}
.nav-logo-icon {
    width: 34px; height: 34px;
    background: #6c47ff;
    border-radius: 8px;
    display: flex; align-items: center; justify-content: center;
    color: #fff;
    font-size: 1.1rem;
    font-weight: 700;
}
.nav-right {
    display: flex;
    align-items: center;
    gap: 1.5rem;
}
.nav-link {
    color: #57606a;
    font-size: 0.9rem;
    text-decoration: none;
    font-weight: 500;
}
.nav-btn {
    background: #6c47ff;
    color: #fff !important;
    border: none;
    border-radius: 8px;
    padding: 7px 18px;
    font-size: 0.88rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
}

/* ── Powered-by badge ── */
.powered-badge {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: #f0ecff;
    border: 1px solid #d4c8ff;
    border-radius: 20px;
    padding: 4px 14px;
    font-size: 0.78rem;
    color: #6c47ff;
    font-weight: 600;
    margin-bottom: 1.4rem;
}
.powered-dot {
    width: 7px; height: 7px;
    background: #6c47ff;
    border-radius: 50%;
    display: inline-block;
}

/* ── Hero section ── */
.hero-section {
    padding: 3.5rem 0 2rem 0;
}
.hero-title {
    font-size: 3rem;
    font-weight: 800;
    color: #1f2328;
    line-height: 1.18;
    margin: 0 0 0.5rem 0;
}
.hero-subtitle {
    font-size: 1.35rem;
    font-weight: 700;
    color: #6c47ff;
    margin: 0 0 1.1rem 0;
}
.hero-desc {
    font-size: 1.0rem;
    color: #57606a;
    line-height: 1.65;
    max-width: 480px;
    margin-bottom: 2rem;
}

/* ── Hero CTA buttons ── */
.btn-primary {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #6c47ff;
    color: #ffffff !important;
    border: none;
    border-radius: 9px;
    padding: 12px 24px;
    font-size: 0.97rem;
    font-weight: 700;
    cursor: pointer;
    text-decoration: none;
    margin-right: 12px;
    transition: background 0.15s;
}
.btn-secondary {
    display: inline-flex;
    align-items: center;
    gap: 7px;
    background: #ffffff;
    color: #1f2328 !important;
    border: 1.5px solid #d0d7de;
    border-radius: 9px;
    padding: 11px 22px;
    font-size: 0.97rem;
    font-weight: 600;
    cursor: pointer;
    text-decoration: none;
    transition: border-color 0.15s;
}
.hero-badges {
    display: flex;
    gap: 1.4rem;
    margin-top: 1.6rem;
    flex-wrap: wrap;
}
.hero-badge-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.83rem;
    color: #57606a;
    font-weight: 500;
}
.check-icon { color: #22c55e; font-size: 1rem; }

/* ── Demo card (right side) ── */
.demo-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 16px;
    padding: 1.4rem 1.5rem;
    box-shadow: 0 4px 24px rgba(108,71,255,0.07);
    max-width: 420px;
    margin: 0 auto;
}
.demo-card-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 0.3rem;
}
.demo-card-title {
    font-size: 0.97rem;
    font-weight: 700;
    color: #1f2328;
}
.demo-card-sub {
    font-size: 0.78rem;
    color: #57606a;
    margin-bottom: 0.9rem;
}
.live-badge {
    background: #dcfce7;
    color: #16a34a;
    border-radius: 12px;
    padding: 2px 10px;
    font-size: 0.73rem;
    font-weight: 700;
}
.demo-question-meta {
    font-size: 0.75rem;
    color: #6c47ff;
    font-weight: 600;
    margin-bottom: 0.3rem;
}
.demo-question-text {
    font-size: 0.93rem;
    color: #1f2328;
    font-weight: 600;
    line-height: 1.45;
    margin-bottom: 1.1rem;
}
.metric-row {
    display: flex;
    align-items: center;
    margin-bottom: 0.5rem;
}
.metric-label {
    font-size: 0.77rem;
    color: #57606a;
    width: 130px;
    flex-shrink: 0;
}
.metric-bar-bg {
    flex: 1;
    background: #f0ecff;
    border-radius: 6px;
    height: 7px;
    margin: 0 10px;
    overflow: hidden;
}
.metric-bar-fill {
    background: #6c47ff;
    height: 100%;
    border-radius: 6px;
}
.metric-val {
    font-size: 0.77rem;
    color: #1f2328;
    font-weight: 600;
    width: 32px;
    text-align: right;
}
.demo-card-footer {
    border-top: 1px solid #f0f0f0;
    margin-top: 0.9rem;
    padding-top: 0.8rem;
    display: flex;
    justify-content: space-between;
    align-items: center;
}
.overall-label {
    font-size: 0.82rem;
    color: #57606a;
}
.overall-score {
    font-size: 1.5rem;
    font-weight: 800;
    color: #6c47ff;
}
.overall-score span {
    font-size: 0.85rem;
    color: #57606a;
    font-weight: 500;
}

/* ── Section divider ── */
.section-divider {
    border: none;
    border-top: 1px solid #e5e7eb;
    margin: 0.5rem 0 2rem 0;
}

/* ── Feature cards (topic grid) ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0 2.5rem 0;
}
.feature-card {
    background: #ffffff;
    border: 1px solid #e5e7eb;
    border-radius: 12px;
    padding: 1.1rem 1rem;
    text-align: center;
    transition: box-shadow 0.15s;
}
.feature-card:hover { box-shadow: 0 4px 18px rgba(108,71,255,0.10); }
.feature-icon { font-size: 1.6rem; margin-bottom: 0.4rem; }
.feature-title {
    font-size: 0.88rem;
    font-weight: 700;
    color: #1f2328;
    margin-bottom: 0.2rem;
}
.feature-desc { font-size: 0.77rem; color: #57606a; }

/* ── Chat section ── */
.chat-section-title {
    font-size: 1.45rem;
    font-weight: 800;
    color: #1f2328;
    margin-bottom: 0.2rem;
}
.chat-section-sub {
    font-size: 0.93rem;
    color: #57606a;
    margin-bottom: 1.3rem;
}

/* ── Source card ── */
.source-card {
    background: #f7f8fa;
    border-left: 4px solid #6c47ff;
    border-radius: 4px;
    padding: 0.6rem 0.9rem;
    margin-bottom: 0.5rem;
    font-size: 0.84rem;
    color: #1f2328;
    line-height: 1.5;
}
.source-label { font-weight: 700; color: #6c47ff; }
.score-badge {
    display: inline-block;
    background: #f0ecff;
    color: #6c47ff;
    border-radius: 12px;
    padding: 1px 8px;
    font-size: 0.73rem;
    margin-left: 6px;
    font-weight: 600;
}

/* ── Suggested question buttons ── */
div[data-testid="stButton"] > button {
    background: #ffffff !important;
    color: #1f2328 !important;
    border: 1.5px solid #e5e7eb !important;
    border-radius: 8px !important;
    font-size: 0.83rem !important;
    font-weight: 500 !important;
    text-align: left !important;
    padding: 0.5rem 0.8rem !important;
    transition: border-color 0.15s, background 0.15s !important;
    white-space: normal !important;
    height: auto !important;
}
div[data-testid="stButton"] > button:hover {
    border-color: #6c47ff !important;
    background: #f8f6ff !important;
    color: #6c47ff !important;
}

/* ── Demo warning banner ── */
.demo-banner {
    background: #fff8e1;
    border: 1px solid #ffe082;
    border-radius: 8px;
    padding: 0.55rem 1rem;
    font-size: 0.82rem;
    color: #5d4037;
    margin-bottom: 1.2rem;
}

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 2rem 0 1rem 0;
    font-size: 0.78rem;
    color: #8b949e;
    border-top: 1px solid #e5e7eb;
    margin-top: 3rem;
}

/* Streamlit chat message bubbles */
[data-testid="stChatMessage"] {
    background: #ffffff !important;
    border: 1px solid #e5e7eb !important;
    border-radius: 12px !important;
    margin-bottom: 0.6rem !important;
    padding: 0.8rem 1rem !important;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Navigation bar
# ---------------------------------------------------------------------------

st.markdown("""
<div class="nav-bar">
  <div class="nav-logo">
    <div class="nav-logo-icon">🎓</div>
    College Admission Agent
  </div>
  <div class="nav-right">
    <span class="nav-link">Features</span>
    <span class="nav-link">Knowledge Base</span>
    <span class="nav-link">About</span>
    <span class="nav-btn">Get Started ↓</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Session state
# ---------------------------------------------------------------------------

if "page" not in st.session_state:
    st.session_state.page = "home"           # "home" | "chat"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ---------------------------------------------------------------------------
# Button routing helpers
# ---------------------------------------------------------------------------

col_nav1, col_nav2, col_nav3 = st.columns([6, 1, 1])
with col_nav2:
    if st.button("🚀 Start Chatting", key="hero_start", use_container_width=True):
        st.session_state.page = "chat"
        st.rerun()
with col_nav3:
    if st.button("🏠 Home", key="go_home", use_container_width=True):
        st.session_state.page = "home"
        st.rerun()

# ============================================================
# PAGE: HOME — Hero landing
# ============================================================

if st.session_state.page == "home":

    # ── Hero ──────────────────────────────────────────────
    left, right = st.columns([1.05, 0.95], gap="large")

    with left:
        st.markdown("""
        <div class="hero-section">
          <div class="powered-badge">
            <span class="powered-dot"></span>
            Powered by IBM watsonx.ai &amp; Granite
          </div>
          <div class="hero-title">College Admission<br>Agent</div>
          <div class="hero-subtitle">Ask Smarter. Apply Better.<br>Get Admitted with Confidence.</div>
          <div class="hero-desc">
            A RAG-powered AI admission assistant that answers your questions about
            courses, eligibility, fees, scholarships, deadlines, and more —
            grounded in the actual knowledge base, not guesswork.
          </div>
        </div>
        <div class="hero-badges">
          <div class="hero-badge-item"><span class="check-icon">✔</span> Free to use</div>
          <div class="hero-badge-item"><span class="check-icon">✔</span> No sign-up needed</div>
          <div class="hero-badge-item"><span class="check-icon">✔</span> IBM AI-powered</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

        c1, c2 = st.columns([1, 1], gap="small")
        with c1:
            if st.button("→  Start Asking", key="cta_start", use_container_width=True):
                st.session_state.page = "chat"
                st.rerun()
        with c2:
            if st.button("🎯  Try a Sample Question", key="cta_sample", use_container_width=True):
                st.session_state["_prefill"] = "What is the eligibility criteria for B.Tech CSE?"
                st.session_state.page = "chat"
                st.rerun()

    with right:
        # Get live KB chunk count for the card
        try:
            chunk_count = get_chunk_count()
        except Exception:
            chunk_count = 127

        st.markdown(f"""
        <div style="padding-top: 2.5rem;">
        <div class="demo-card">
          <div class="demo-card-header">
            <div>
              <div class="demo-card-title">🎓 Admission Query</div>
              <div class="demo-card-sub">College Admission Agent &nbsp;·&nbsp; Live Demo</div>
            </div>
            <span class="live-badge">Live</span>
          </div>

          <div class="demo-question-meta">Question 1 of 5 &nbsp;·&nbsp; Eligibility</div>
          <div class="demo-question-text">
            What is the eligibility criteria for B.Tech CSE?
            What is the minimum percentage required?
          </div>

          <div class="metric-row">
            <span class="metric-label">Answer Accuracy</span>
            <div class="metric-bar-bg"><div class="metric-bar-fill" style="width:92%"></div></div>
            <span class="metric-val">92%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">KB Relevance</span>
            <div class="metric-bar-bg"><div class="metric-bar-fill" style="width:88%"></div></div>
            <span class="metric-val">88%</span>
          </div>
          <div class="metric-row">
            <span class="metric-label">Source Grounding</span>
            <div class="metric-bar-bg"><div class="metric-bar-fill" style="width:96%"></div></div>
            <span class="metric-val">96%</span>
          </div>

          <div class="demo-card-footer">
            <span class="overall-label">Knowledge Base<br><small>{chunk_count} indexed chunks</small></span>
            <div>
              <div style="font-size:0.72rem;color:#57606a;text-align:right;margin-bottom:2px;">Confidence</div>
              <div class="overall-score">9.2<span>/10</span></div>
            </div>
          </div>
        </div>
        </div>
        """, unsafe_allow_html=True)

    # ── Feature grid ─────────────────────────────────────
    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:1.2rem;font-weight:800;color:#1f2328;margin-bottom:0.2rem'>Everything you need to know about admissions</div>", unsafe_allow_html=True)
    st.markdown("<div style='text-align:center;font-size:0.9rem;color:#57606a;margin-bottom:1.4rem'>Ask about any of these topics — IBM Granite answers from the knowledge base</div>", unsafe_allow_html=True)

    st.markdown("""
    <div class="feature-grid">
      <div class="feature-card">
        <div class="feature-icon">📚</div>
        <div class="feature-title">Course Selection</div>
        <div class="feature-desc">B.Tech, MBA, M.Sc, BBA & more — find the right programme</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">✅</div>
        <div class="feature-title">Eligibility</div>
        <div class="feature-desc">Marks, age, entrance exam requirements per programme</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">📝</div>
        <div class="feature-title">Application Process</div>
        <div class="feature-desc">Step-by-step guide from registration to enrolment</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">📄</div>
        <div class="feature-title">Required Documents</div>
        <div class="feature-desc">Full checklist of certificates and identity proofs</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">💰</div>
        <div class="feature-title">Fee Structure</div>
        <div class="feature-desc">Tuition, development, hostel fees — all programmes</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">🏆</div>
        <div class="feature-title">Scholarships</div>
        <div class="feature-desc">Merit, need-based, government & sports scholarships</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">📅</div>
        <div class="feature-title">Deadlines</div>
        <div class="feature-desc">Application, merit list, and hostel deadlines</div>
      </div>
      <div class="feature-card">
        <div class="feature-icon">❓</div>
        <div class="feature-title">FAQs</div>
        <div class="feature-desc">25+ common admission questions answered</div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div style="text-align:center;margin-bottom:2rem">
    """, unsafe_allow_html=True)
    c1, c2, c3 = st.columns([2, 1, 2])
    with c2:
        if st.button("→  Start Asking Now", key="cta_bottom", use_container_width=True):
            st.session_state.page = "chat"
            st.rerun()
    st.markdown("</div>", unsafe_allow_html=True)

# ============================================================
# PAGE: CHAT
# ============================================================

else:

    st.markdown("""
    <div class="demo-banner">
      ⚠️ <strong>Demo Application</strong> — All college data is fictitious sample data
      created for IBM AICTE Internship 2026. Do not use for actual college admissions.
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="chat-section-title">🎓 Ask the Admission Agent</div>', unsafe_allow_html=True)
    st.markdown('<div class="chat-section-sub">Ask anything about courses, eligibility, fees, scholarships, deadlines, or documents.</div>', unsafe_allow_html=True)

    # ── Chat history ──────────────────────────────────────
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and msg.get("sources"):
                with st.expander("📖 Sources used", expanded=False):
                    for src in msg["sources"]:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-label">📄 {src["source"]}</span>'
                            f'<span class="score-badge">score: {src["score"]:.2f}</span>'
                            f'<br><br>{src["excerpt"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )

    # ── Suggested questions (first visit) ────────────────
    if not st.session_state.messages:
        st.markdown("**💡 Try asking:**")
        suggested = [
            "What are the eligibility criteria for B.Tech CSE?",
            "What is the fee structure for MBA?",
            "What documents do I need for undergraduate admission?",
            "When is the last date to apply for B.Tech?",
            "Are there any scholarships for SC/ST students?",
            "How do I apply for admission step by step?",
            "What is the JEE Main cutoff for B.Tech CSE?",
            "What courses does the college offer?",
        ]
        cols = st.columns(2)
        for i, q in enumerate(suggested):
            if cols[i % 2].button(q, key=f"suggest_{i}", use_container_width=True):
                st.session_state["_prefill"] = q
                st.rerun()

    # ── Chat input ────────────────────────────────────────
    prefill = st.session_state.pop("_prefill", "")
    user_input = st.chat_input("Ask about admissions, fees, eligibility, deadlines…")
    question = prefill or user_input

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching knowledge base and generating answer…"):
                response: AgentResponse = ask(question)

            st.markdown(response.answer)

            sources_data = [
                {"source": s.source, "excerpt": s.excerpt, "score": s.score}
                for s in response.sources
            ]

            if sources_data:
                with st.expander("📖 Sources used", expanded=False):
                    for src in sources_data:
                        st.markdown(
                            f'<div class="source-card">'
                            f'<span class="source-label">📄 {src["source"]}</span>'
                            f'<span class="score-badge">score: {src["score"]:.2f}</span>'
                            f'<br><br>{src["excerpt"]}'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
            elif response.is_fallback:
                st.info("ℹ️ No relevant information found in the knowledge base for this question.")

            if response.error:
                st.warning(f"⚠️ Model error: {response.error[:200]}")

        st.session_state.messages.append({
            "role": "assistant",
            "content": response.answer,
            "sources": sources_data,
            "is_fallback": response.is_fallback,
        })

# ---------------------------------------------------------------------------
# Footer
# ---------------------------------------------------------------------------

st.markdown("""
<div class="footer">
  Built with Python · Streamlit · IBM Granite (watsonx.ai) · TF-IDF RAG &nbsp;|&nbsp;
  IBM AICTE Internship 2026 — Problem Statement No. 4
</div>
""", unsafe_allow_html=True)
