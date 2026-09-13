import tempfile
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

from agents.ats_agent import analyze_resume_ats
from agents.application_agent import generate_application_package
from agents import matching_agent
from agents.discovery_agent import discover_opportunities
from agents.profile_agent import analyze_candidate_profile
from agents.resume_optimizer import optimize_resume
from services.document_service import extract_text_from_resume
from services.resume_document_service import generate_resume_docx, generate_resume_pdf

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

if "match_results" not in st.session_state:
    st.session_state["match_results"] = []

if "match_attempted" not in st.session_state:
    st.session_state["match_attempted"] = False

if "resume_text" not in st.session_state:
    st.session_state["resume_text"] = ""

if "selected_opportunity_id" not in st.session_state:
    st.session_state["selected_opportunity_id"] = None

if "ats_report" not in st.session_state:
    st.session_state["ats_report"] = None

if "resume_optimization" not in st.session_state:
    st.session_state["resume_optimization"] = None

if "application_package" not in st.session_state:
    st.session_state["application_package"] = None


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

                resume_text = extract_text_from_resume(
                    temp_path
                )

                candidate = analyze_candidate_profile(
                    resume_text
                )

            st.session_state[
                "candidate_profile"
            ] = candidate

            # Clear old opportunities because the candidate
            # profile has changed.
            st.session_state[
                "opportunities"
            ] = []

            st.session_state[
                "match_results"
            ] = []

            st.session_state[
                "match_attempted"
            ] = False

            st.session_state[
                "resume_text"
            ] = resume_text

            st.session_state[
                "selected_opportunity_id"
            ] = None

            st.session_state[
                "ats_report"
            ] = None

            st.session_state[
                "resume_optimization"
            ] = None

            st.session_state[
                "application_package"
            ] = None

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

            st.session_state[
                "match_results"
            ] = []

            st.session_state[
                "match_attempted"
            ] = False

            st.session_state[
                "selected_opportunity_id"
            ] = None

            st.session_state[
                "ats_report"
            ] = None

            st.session_state[
                "resume_optimization"
            ] = None

            st.session_state[
                "application_package"
            ] = None

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
# OPPORTUNITY MATCHING
# =========================================================
match_results = st.session_state.get(
    "match_results",
    [],
)

render_html(
    """
    <div class="section-title">
    🎯 Match Opportunities
    </div>
    """
)

can_match = candidate is not None and bool(opportunities)

if candidate is None:
    st.info("Analyze your resume before matching opportunities.")
elif not opportunities:
    st.info("Discover opportunities before matching them to your profile.")

if st.button(
    "🎯 Match Opportunities",
    use_container_width=True,
    disabled=not can_match,
):
    try:
        with st.spinner("Analyzing your opportunity matches..."):
            match_function = getattr(
                matching_agent,
                "match_opportunities",
                matching_agent.evaluate_opportunities,
            )
            match_results = match_function(
                candidate,
                opportunities,
            )
        st.session_state["match_results"] = match_results
        st.session_state["match_attempted"] = True
    except Exception:
        st.session_state["match_results"] = []
        st.session_state["match_attempted"] = True
        st.error(
            "Unable to match opportunities right now. "
            "Please try again later."
        )

if match_results:
    opportunity_by_id = {
        opportunity.id: opportunity
        for opportunity in opportunities
    }
    sorted_matches = sorted(
        match_results,
        key=lambda result: result.match_score,
        reverse=True,
    )

    for index, match in enumerate(sorted_matches, start=1):
        opportunity = opportunity_by_id.get(match.opportunity_id)
        if opportunity is None:
            continue

        score = float(match.match_score)
        if score >= 90:
            score_label = "Excellent"
        elif score >= 75:
            score_label = "Strong"
        elif score >= 60:
            score_label = "Moderate"
        else:
            score_label = "Low"

        render_html(
            f"""
            <div class="opportunity-card">
            <div class="opportunity-title">
            {index}. {opportunity.title}
            </div>
            <div class="opportunity-org">
            {opportunity.organization}
            </div>
            </div>
            """
        )

        score_col, eligibility_col = st.columns(2)
        with score_col:
            st.metric(
                "Match score",
                f"{score:.0f}/100",
                score_label,
            )
        with eligibility_col:
            st.metric(
                "Eligibility",
                "Eligible" if match.eligible else "Not Eligible",
            )

        detail_col, reason_col = st.columns(2)
        with detail_col:
            st.markdown("**Strengths**")
            if match.strengths:
                for strength in match.strengths:
                    st.markdown(f"- {strength}")
            else:
                st.caption("No strengths reported.")

            st.markdown("**Gaps**")
            if match.gaps:
                for gap in match.gaps:
                    st.markdown(f"- {gap}")
            else:
                st.caption("No gaps reported.")

        with reason_col:
            st.markdown("**Reasons**")
            if match.reasons:
                for reason in match.reasons:
                    st.markdown(f"- {reason}")
            else:
                st.caption("No reasons reported.")

        st.markdown("---")
elif can_match and st.session_state.get("match_attempted", False):
    if st.session_state["match_results"] == []:
        st.info("No matching opportunities found.")


# =========================================================
# ATS INTELLIGENCE
# =========================================================
ats_report = st.session_state.get(
    "ats_report",
)
resume_text = st.session_state.get(
    "resume_text",
    "",
)

render_html(
    """
    <div class="section-title">
    📊 ATS Intelligence
    </div>
    """
)

match_by_id = {
    match.opportunity_id: match
    for match in match_results
}
if match_results:
    selection_order = [
        match.opportunity_id
        for match in sorted(
            match_results,
            key=lambda result: result.match_score,
            reverse=True,
        )
        if match.opportunity_id in {
            opportunity.id for opportunity in opportunities
        }
    ]
else:
    selection_order = [opportunity.id for opportunity in opportunities]

opportunity_by_id = {
    opportunity.id: opportunity
    for opportunity in opportunities
}
selection_order = [
    opportunity_id
    for opportunity_id in selection_order
    if opportunity_id in opportunity_by_id
]

if selection_order:
    previous_selected_id = st.session_state.get("selected_opportunity_id")
    selected_id = st.session_state.get("selected_opportunity_id")
    if selected_id not in selection_order:
        selected_id = selection_order[0]

    selected_id = st.selectbox(
        "Select an opportunity for ATS analysis",
        options=selection_order,
        index=selection_order.index(selected_id),
        format_func=lambda opportunity_id: (
            f"{opportunity_by_id[opportunity_id].title} - "
            f"{opportunity_by_id[opportunity_id].organization}"
        ),
    )
    st.session_state["selected_opportunity_id"] = selected_id
    if previous_selected_id != selected_id:
        st.session_state["resume_optimization"] = None
        st.session_state["application_package"] = None
else:
    st.info("Select an opportunity after discovering and matching opportunities.")

selected_opportunity_id = st.session_state.get(
    "selected_opportunity_id"
)
can_analyze_ats = bool(
    candidate is not None
    and resume_text
    and selected_opportunity_id in opportunity_by_id
)

if st.button(
    "📊 Analyze ATS",
    use_container_width=True,
    disabled=not can_analyze_ats,
):
    selected_opportunity = opportunity_by_id[selected_opportunity_id]
    try:
        with st.spinner("Analyzing resume compatibility..."):
            ats_report = analyze_resume_ats(
                candidate,
                selected_opportunity,
                resume_text,
            )
        st.session_state["ats_report"] = ats_report
    except Exception:
        st.session_state["ats_report"] = None
        st.error(
            "Unable to analyze ATS compatibility right now. "
            "Please try again later."
        )

if ats_report is not None:
    st.metric("ATS Score", f"{ats_report.ats_score:.0f} / 100")
    ats_columns = st.columns(2)
    report_sections = (
        ("Strengths", ats_report.strengths),
        ("Issues", ats_report.issues),
        ("Matched Keywords", ats_report.matched_keywords),
        ("Missing Keywords", ats_report.missing_keywords),
        ("Recommendations", ats_report.recommendations),
    )
    for index, (label, items) in enumerate(report_sections):
        with ats_columns[index % 2]:
            st.markdown(f"**{label}**")
            if items:
                for item in items:
                    st.markdown(f"- {item}")
            else:
                st.caption(f"No {label.lower()} reported.")
else:
    st.info(
        "Select an opportunity and click Analyze ATS to review resume alignment."
    )


# =========================================================
# RESUME OPTIMIZATION
# =========================================================
resume_optimization = st.session_state.get(
    "resume_optimization",
)
selected_opportunity = opportunity_by_id.get(selected_opportunity_id)
can_optimize_resume = bool(
    candidate is not None
    and resume_text
    and selected_opportunity is not None
    and ats_report is not None
)

if can_optimize_resume:
    render_html(
        """
        <div class="section-title">
        ✨ Resume Optimization
        </div>
        """
    )

    if st.button(
        "✨ Optimize Resume",
        use_container_width=True,
        disabled=not can_optimize_resume,
    ):
        try:
            with st.spinner("Optimizing your resume..."):
                resume_optimization = optimize_resume(
                    candidate,
                    selected_opportunity,
                    resume_text,
                    ats_report,
                )
            st.session_state["resume_optimization"] = resume_optimization
            st.session_state["application_package"] = None
        except Exception:
            st.session_state["resume_optimization"] = None
            st.error(
                "Unable to optimize your resume right now. "
                "Please try again later."
            )

    if resume_optimization is not None:
        st.markdown("### Optimized Resume")
        st.text_area(
            "Optimized Resume",
            value=resume_optimization.optimized_resume,
            height=420,
            disabled=True,
            label_visibility="collapsed",
        )
        st.download_button(
            "⬇️ Download Optimized Resume",
            data=resume_optimization.optimized_resume,
            file_name="optimized_resume.txt",
            mime="text/plain",
            use_container_width=True,
        )

        optimization_columns = st.columns(2)
        optimization_sections = (
            ("Changed Sections", resume_optimization.changed_sections),
            ("Added Keywords", resume_optimization.added_keywords),
            ("Removed / Weak Content", resume_optimization.removed_or_weak_content),
            ("Recommendations", resume_optimization.recommendations),
        )
        for index, (label, items) in enumerate(optimization_sections):
            with optimization_columns[index % 2]:
                st.markdown(f"**{label}**")
                if items:
                    for item in items:
                        st.markdown(f"- {item}")
                else:
                    st.caption(f"No {label.lower()} reported.")
    else:
        st.info("Click Optimize Resume to generate an optimized resume.")
else:
    missing_requirements = []
    if candidate is None:
        missing_requirements.append("candidate profile")
    if not resume_text:
        missing_requirements.append("resume text")
    if selected_opportunity is None:
        missing_requirements.append("selected opportunity")
    if ats_report is None:
        missing_requirements.append("ATS report")
    st.info(
        "Resume optimization is unavailable until you provide: "
        + ", ".join(missing_requirements)
        + "."
    )


# =========================================================
# APPLICATION PACKAGE
# =========================================================
application_package = st.session_state.get(
    "application_package",
)
can_prepare_application = bool(
    candidate is not None
    and resume_text
    and selected_opportunity is not None
    and resume_optimization is not None
    and ats_report is not None
)

if can_prepare_application:
    render_html(
        """
        <div class="section-title">
        📦 Application Package
        </div>
        """
    )

    if st.button(
        "📦 Prepare Application",
        use_container_width=True,
    ):
        try:
            with st.spinner("Preparing your application package..."):
                # The checked-in AP-006 agent currently requires the ATS report
                # alongside the optimized resume.
                application_package = generate_application_package(
                    candidate,
                    selected_opportunity,
                    resume_optimization,
                    ats_report,
                )
            st.session_state["application_package"] = application_package
        except Exception:
            st.session_state["application_package"] = None
            st.error(
                "Unable to prepare your application package right now. "
                "Please try again later."
            )

    if application_package is not None:
        st.markdown("### Cover Letter")
        cover_letter = application_package.cover_letter or ""
        st.text_area(
            "Cover Letter",
            value=cover_letter,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
        st.download_button(
            "⬇️ Download Cover Letter",
            data=cover_letter,
            file_name="cover_letter.txt",
            mime="text/plain",
        )

        st.markdown("### SOP")
        sop = application_package.sop or ""
        st.text_area(
            "SOP",
            value=sop,
            height=260,
            disabled=True,
            label_visibility="collapsed",
        )
        st.download_button(
            "⬇️ Download SOP",
            data=sop,
            file_name="statement_of_purpose.txt",
            mime="text/plain",
        )

        st.markdown("### Application Answers")
        application_answers = "\n\n".join(
            application_package.application_answers
        )
        if application_answers:
            st.text_area(
                "Application Answers",
                value=application_answers,
                height=220,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.caption("No application answers were generated.")
        st.download_button(
            "⬇️ Download Application Answers",
            data=application_answers,
            file_name="application_answers.txt",
            mime="text/plain",
        )

        package_columns = st.columns(2)
        with package_columns[0]:
            st.markdown("### Required Documents")
            if application_package.required_documents:
                for document in application_package.required_documents:
                    st.markdown(f"- {document}")
            else:
                st.caption("No required documents listed.")

        with package_columns[1]:
            st.markdown("### Application Checklist")
            if application_package.checklist:
                for index, item in enumerate(application_package.checklist):
                    st.checkbox(
                        item,
                        key=f"application_checklist_{index}",
                    )
            else:
                st.caption("No checklist items listed.")

        st.markdown("### Resume Downloads")
        optimized_text = resume_optimization.optimized_resume
        st.download_button(
            "⬇️ Download Optimized Resume",
            data=optimized_text,
            file_name="optimized_resume.txt",
            mime="text/plain",
        )

        document_downloads = st.columns(2)
        for column, generator, suffix, label, mime in (
            (
                document_downloads[0],
                generate_resume_docx,
                ".docx",
                "⬇️ Download Optimized Resume DOCX",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            ),
            (
                document_downloads[1],
                generate_resume_pdf,
                ".pdf",
                "⬇️ Download Optimized Resume PDF",
                "application/pdf",
            ),
        ):
            try:
                document_path = None
                with tempfile.NamedTemporaryFile(
                    delete=False,
                    suffix=suffix,
                ) as document_file:
                    document_path = document_file.name
                generator(optimized_text, document_path)
                document_data = Path(document_path).read_bytes()
                column.download_button(
                    label,
                    data=document_data,
                    file_name=f"optimized_resume{suffix}",
                    mime=mime,
                )
            except Exception:
                column.warning("This document format is temporarily unavailable.")
            finally:
                if document_path:
                    Path(document_path).unlink(missing_ok=True)
    else:
        st.info("Click Prepare Application to generate your application package.")
else:
    missing_requirements = []
    if candidate is None:
        missing_requirements.append("candidate profile")
    if not resume_text:
        missing_requirements.append("resume text")
    if selected_opportunity is None:
        missing_requirements.append("selected opportunity")
    if resume_optimization is None:
        missing_requirements.append("optimized resume")
    if ats_report is None:
        missing_requirements.append("ATS report required by the application agent")
    st.info(
        "Application package is unavailable until you complete: "
        + ", ".join(missing_requirements)
        + "."
    )


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