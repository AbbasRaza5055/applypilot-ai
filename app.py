import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents.discovery_agent import discover_opportunities
from agents.profile_agent import process_resume

load_dotenv()

# =========================================================
# Helper: render HTML safely
# =========================================================
def render_html(html: str) -> None:
    """
    Render HTML inside Streamlit without accidental Markdown
    code-block formatting.
    """
    cleaned_html = "\n".join(
        line.strip()
        for line in html.strip().splitlines()
        if line.strip()
    )

    st.markdown(
        cleaned_html,
        unsafe_allow_html=True,
    )


# =========================================================
# Page configuration
# =========================================================
st.set_page_config(
    page_title="ApplyPilot AI",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)


# =========================================================
# Global CSS
# =========================================================
st.markdown(
    """
<style>
    /* =====================================================
       APP BACKGROUND
    ===================================================== */
    .stApp {
        background:
            radial-gradient(
                circle at 10% 10%,
                rgba(99, 102, 241, 0.12),
                transparent 30%
            ),
            radial-gradient(
                circle at 90% 20%,
                rgba(14, 165, 233, 0.10),
                transparent 28%
            ),
            #0b1020;

        color: #f8fafc;
    }


    /* =====================================================
       MAIN CONTAINER
    ===================================================== */
    .block-container {
        max-width: 1250px;
        padding-top: 2rem;
        padding-bottom: 3rem;
    }


    /* =====================================================
       SIDEBAR
    ===================================================== */
    section[data-testid="stSidebar"] {
        background:
            linear-gradient(
                180deg,
                #0f172a 0%,
                #111827 100%
            );

        border-right: 1px solid rgba(255, 255, 255, 0.08);
    }

    .sidebar-brand {
        padding: 1rem 0 1.5rem 0;
    }

    .sidebar-brand h2 {
        margin: 0;
        font-size: 1.35rem;
        color: #ffffff;
    }

    .sidebar-brand p {
        color: #94a3b8;
        margin-top: 0.35rem;
        font-size: 0.85rem;
    }

    .workflow-item {
        padding: 0.65rem 0.75rem;
        margin: 0.3rem 0;

        border-radius: 10px;

        color: #cbd5e1;

        background: rgba(255, 255, 255, 0.025);

        border: 1px solid rgba(255, 255, 255, 0.05);

        transition: all 0.2s ease;
    }

    .workflow-item.active {
        background:
            linear-gradient(
                90deg,
                rgba(99, 102, 241, 0.20),
                rgba(14, 165, 233, 0.10)
            );

        border-color: rgba(99, 102, 241, 0.35);

        color: #ffffff;
    }

    .workflow-number {
        display: inline-flex;

        align-items: center;
        justify-content: center;

        width: 24px;
        height: 24px;

        border-radius: 50%;

        margin-right: 8px;

        background: rgba(255, 255, 255, 0.08);

        color: #cbd5e1;

        font-size: 0.78rem;
    }


    /* =====================================================
       HERO
    ===================================================== */
    .hero {
        padding: 2rem 0 1rem 0;
    }

    .eyebrow {
        color: #818cf8;

        font-weight: 700;

        letter-spacing: 0.10em;

        text-transform: uppercase;

        font-size: 0.78rem;

        margin-bottom: 0.8rem;
    }

    .hero h1 {
        font-size: clamp(2.5rem, 5vw, 4.5rem);

        line-height: 1.02;

        margin: 0;

        font-weight: 800;

        letter-spacing: -0.04em;

        color: #ffffff;
    }

    .hero h1 span {
        background:
            linear-gradient(
                90deg,
                #818cf8,
                #22d3ee
            );

        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }

    .hero p {
        max-width: 760px;

        color: #94a3b8;

        font-size: 1.08rem;

        line-height: 1.7;

        margin-top: 1.2rem;
    }


    /* =====================================================
       PILLS
    ===================================================== */
    .pill-row {
        display: flex;

        gap: 0.6rem;

        flex-wrap: wrap;

        margin: 1.25rem 0 1.8rem 0;
    }

    .pill {
        display: inline-block;

        padding: 0.42rem 0.8rem;

        border-radius: 999px;

        background: rgba(255, 255, 255, 0.05);

        border: 1px solid rgba(255, 255, 255, 0.08);

        color: #cbd5e1;

        font-size: 0.82rem;
    }


    /* =====================================================
       UPLOAD CARD
    ===================================================== */
    .upload-card {
        padding: 1.4rem;

        border-radius: 20px;

        background:
            linear-gradient(
                145deg,
                rgba(30, 41, 59, 0.88),
                rgba(15, 23, 42, 0.92)
            );

        border: 1px solid rgba(129, 140, 248, 0.20);

        box-shadow:
            0 20px 50px rgba(0, 0, 0, 0.20);

        margin-top: 1rem;
    }

    .upload-title {
        font-size: 1.1rem;

        font-weight: 700;

        margin-bottom: 0.3rem;

        color: #ffffff;
    }

    .upload-subtitle {
        color: #94a3b8;

        font-size: 0.9rem;

        line-height: 1.6;
    }


    /* =====================================================
       FEATURE CARDS
    ===================================================== */
    .feature-card {
        padding: 1.25rem;

        border-radius: 16px;

        background: rgba(15, 23, 42, 0.75);

        border: 1px solid rgba(255, 255, 255, 0.07);

        min-height: 180px;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.12);

        transition:
            transform 0.2s ease,
            border-color 0.2s ease;
    }

    .feature-card:hover {
        transform: translateY(-3px);

        border-color:
            rgba(129, 140, 248, 0.25);
    }

    .feature-icon {
        font-size: 1.6rem;

        margin-bottom: 0.65rem;
    }

    .feature-title {
        font-weight: 700;

        margin-bottom: 0.4rem;

        color: #ffffff;
    }

    .feature-text {
        color: #94a3b8;

        font-size: 0.86rem;

        line-height: 1.6;
    }


    /* =====================================================
       SECTION TITLES
    ===================================================== */
    .section-title {
        font-size: 1.3rem;

        font-weight: 750;

        margin: 2.2rem 0 1rem 0;

        color: #ffffff;
    }


    /* =====================================================
       OPPORTUNITY CARDS
    ===================================================== */
    .opportunity-card {
        padding: 1.3rem;

        border-radius: 18px;

        background:
            linear-gradient(
                145deg,
                rgba(15, 23, 42, 0.95),
                rgba(30, 41, 59, 0.80)
            );

        border: 1px solid
            rgba(255, 255, 255, 0.08);

        margin-bottom: 1rem;

        box-shadow:
            0 12px 30px rgba(0, 0, 0, 0.12);
    }

    .opportunity-title {
        color: #ffffff;

        font-size: 1.15rem;

        font-weight: 750;

        margin-bottom: 0.25rem;
    }

    .opportunity-org {
        color: #818cf8;

        font-weight: 600;

        margin-bottom: 0.6rem;
    }

    .opportunity-meta {
        color: #94a3b8;

        font-size: 0.84rem;

        margin-bottom: 0.7rem;
    }

    .opportunity-description {
        color: #cbd5e1;

        font-size: 0.9rem;

        line-height: 1.65;
    }

    .tag {
        display: inline-block;

        margin-right: 0.4rem;
        margin-bottom: 0.35rem;

        padding: 0.3rem 0.6rem;

        border-radius: 999px;

        background: rgba(99, 102, 241, 0.10);

        border: 1px solid
            rgba(99, 102, 241, 0.20);

        color: #c7d2fe;

        font-size: 0.75rem;
    }


    /* =====================================================
       BUTTONS
    ===================================================== */
    .stButton > button {
        border-radius: 12px;

        border: 1px solid
            rgba(129, 140, 248, 0.35);

        background:
            linear-gradient(
                90deg,
                #6366f1,
                #0ea5e9
            );

        color: white;

        font-weight: 700;

        padding: 0.7rem 1rem;

        transition: 0.2s ease;
    }

    .stButton > button:hover {
        border-color:
            rgba(255, 255, 255, 0.4);

        transform: translateY(-1px);
    }


    /* =====================================================
       FILE UPLOADER
    ===================================================== */
    div[data-testid="stFileUploader"] {
        margin-top: 0.5rem;
    }


    /* =====================================================
       FOOTER
    ===================================================== */
    .footer {
        text-align: center;

        color: #64748b;

        font-size: 0.78rem;

        margin-top: 3rem;

        padding-top: 1rem;

        border-top:
            1px solid
            rgba(255, 255, 255, 0.06);
    }
</style>
""",
    unsafe_allow_html=True,
)


# =========================================================
# SIDEBAR
# =========================================================
with st.sidebar:

    render_html(
        """
        <div class="sidebar-brand">
        <h2>🚀 ApplyPilot AI</h2>
        <p>Your opportunity copilot</p>
        </div>
        """
    )

    st.markdown("### Workflow")

    steps = [
        ("01", "Profile"),
        ("02", "Discover"),
        ("03", "Match"),
        ("04", "ATS Check"),
        ("05", "Optimize"),
        ("06", "Apply"),
    ]

    for number, label in steps:

        active_class = (
            "active"
            if number in {"01", "02"}
            else ""
        )

        render_html(
            f"""
            <div class="workflow-item {active_class}">
            <span class="workflow-number">
            {number}
            </span>
            {label}
            </div>
            """
        )

    st.markdown("---")

    st.caption(
        "AI-powered • ATS-ready • Opportunity focused"
    )


# =========================================================
# HERO
# =========================================================
render_html(
    """
    <div class="hero">

    <div class="eyebrow">
    AI Career Copilot
    </div>

    <h1>
    Your career journey,
    <span>on autopilot.</span>
    </h1>

    <p>
    Discover the right opportunities, understand your match,
    optimize your resume, and prepare your application —
    all from one intelligent workspace.
    </p>

    <div class="pill-row">

    <div class="pill">
    ✦ AI Powered
    </div>

    <div class="pill">
    ✓ ATS Ready
    </div>

    <div class="pill">
    ⚡ One Workspace
    </div>

    <div class="pill">
    🎯 Opportunity Focused
    </div>

    </div>

    </div>
    """
)


# =========================================================
# INITIALIZE SESSION STATE
# =========================================================
if "candidate_profile" not in st.session_state:
    st.session_state["candidate_profile"] = None

if "opportunities" not in st.session_state:
    st.session_state["opportunities"] = []


# =========================================================
# RESUME UPLOAD SECTION
# =========================================================
render_html(
    """
    <div class="upload-card">

    <div class="upload-title">
    Start with your resume
    </div>

    <div class="upload-subtitle">
    Upload your CV in PDF or DOCX format.
    ApplyPilot will build your candidate profile automatically.
    </div>

    </div>
    """
)


uploaded_file = st.file_uploader(
    "Upload Resume",
    type=["pdf", "docx"],
    label_visibility="collapsed",
)


# =========================================================
# PROFILE ANALYSIS
# =========================================================
if uploaded_file is not None:

    st.success(
        f"Resume ready: {uploaded_file.name}"
    )

    if st.button(
        "🚀 Analyze My Resume",
        use_container_width=True,
    ):

        temp_path = None

        try:

            suffix = Path(
                uploaded_file.name
            ).suffix

            with tempfile.NamedTemporaryFile(
                delete=False,
                suffix=suffix,
            ) as temp_file:

                temp_file.write(
                    uploaded_file.getbuffer()
                )

                temp_path = temp_file.name

            with st.spinner(
                "Analyzing your resume..."
            ):

                candidate = process_resume(
                    temp_path
                )

            st.session_state[
                "candidate_profile"
            ] = candidate

            # Clear old opportunities because the candidate
            # profile has changed.
            st.session_state[
                "opportunities"
            ] = []

            st.success(
                "Resume analyzed successfully!"
            )

        except Exception as exc:

            st.error(
                f"Unable to analyze the resume: {exc}"
            )

        finally:

            if temp_path:

                try:
                    Path(
                        temp_path
                    ).unlink(
                        missing_ok=True
                    )

                except Exception:
                    pass


# =========================================================
# CANDIDATE PROFILE
# =========================================================
candidate = st.session_state.get(
    "candidate_profile"
)

if candidate is not None:

    render_html(
        """
        <div class="section-title">
        👤 Your Candidate Profile
        </div>
        """
    )

    with st.expander(
        "View extracted profile",
        expanded=True,
    ):
        st.json(
            candidate.model_dump()
        )


# =========================================================
# OPPORTUNITY DISCOVERY
# =========================================================
if candidate is not None:

    render_html(
        """
        <div class="section-title">
        🔎 Discover Opportunities
        </div>
        """
    )

    st.caption(
        "Find opportunities based on your skills, "
        "education, interests, and profile."
    )

    if st.button(
        "✨ Find Opportunities",
        use_container_width=True,
    ):

        try:

            with st.spinner(
                "Searching for relevant opportunities..."
            ):

                opportunities = discover_opportunities(
                    candidate,
                    max_results=10,
                )

            st.session_state[
                "opportunities"
            ] = opportunities

            if not opportunities:

                st.info(
                    "No matching opportunities were found."
                )

            else:

                st.success(
                    f"Found {len(opportunities)} "
                    "relevant opportunities."
                )

        except Exception as exc:

            st.error(
                f"Unable to discover opportunities: {exc}"
            )


# =========================================================
# OPPORTUNITY RESULTS
# =========================================================
opportunities = st.session_state.get(
    "opportunities",
    [],
)


if opportunities:

    render_html(
        """
        <div class="section-title">
        🎯 Recommended Opportunities
        </div>
        """
    )

    for index, opportunity in enumerate(
        opportunities,
        start=1,
    ):

        render_html(
            f"""
            <div class="opportunity-card">

            <div class="opportunity-title">
            {index}. {opportunity.title}
            </div>

            <div class="opportunity-org">
            {opportunity.organization}
            </div>

            <div class="opportunity-meta">
            {opportunity.type.title()}
            {" • 📍 " + opportunity.location if opportunity.location else ""}
            {" • ⏰ Deadline: " + opportunity.deadline if opportunity.deadline else ""}
            </div>

            <div class="opportunity-description">
            {opportunity.description}
            </div>

            </div>
            """
        )

        if opportunity.requirements:

            st.markdown(
                "**Key requirements**"
            )

            tags = " ".join(
                f'<span class="tag">{requirement}</span>'
                for requirement
                in opportunity.requirements[:8]
            )

            render_html(
                f"""
                <div>
                {tags}
                </div>
                """
            )

        if opportunity.url:

            st.link_button(
                "🔗 View Opportunity",
                opportunity.url,
            )

        st.markdown("---")


# =========================================================
# FEATURE SECTION
# =========================================================
render_html(
    """
    <div class="section-title">
    What ApplyPilot does
    </div>
    """
)


col1, col2, col3 = st.columns(3)


with col1:

    render_html(
        """
        <div class="feature-card">

        <div class="feature-icon">
        🔎
        </div>

        <div class="feature-title">
        Discover
        </div>

        <div class="feature-text">
        Find relevant jobs, internships,
        scholarships, hackathons, and fellowships.
        </div>

        </div>
        """
    )


with col2:

    render_html(
        """
        <div class="feature-card">

        <div class="feature-icon">
        🎯
        </div>

        <div class="feature-title">
        Match & Analyze
        </div>

        <div class="feature-text">
        Understand your eligibility, opportunity match,
        and ATS readiness.
        </div>

        </div>
        """
    )


with col3:

    render_html(
        """
        <div class="feature-card">

        <div class="feature-icon">
        ⚡
        </div>

        <div class="feature-title">
        Optimize & Apply
        </div>

        <div class="feature-text">
        Improve your resume and prepare the documents
        needed for your application.
        </div>

        </div>
        """
    )


# =========================================================
# FOOTER
# =========================================================
render_html(
    """
    <div class="footer">
    ApplyPilot AI • One Profile. Every Opportunity.
    Zero Missed Deadlines.
    </div>
    """
)