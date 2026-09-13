from models import (
    ApplicationPackage,
    ATSReport,
    CandidateProfile,
    MatchResult,
    Opportunity,
)

from models.resume_optimization import ResumeOptimizationResult


def test_candidate_profile():
    profile = CandidateProfile(
        name="Test User",
        skills=["Python", "Machine Learning"],
    )

    assert profile.name == "Test User"
    assert "Python" in profile.skills


def test_opportunity():
    opportunity = Opportunity(
        id="opp_001",
        title="AI/ML Intern",
        organization="Example Company",
        type="internship",
        description="AI/ML internship opportunity",
        requirements=["Python", "Machine Learning"],
        url="https://example.com",
    )

    assert opportunity.id == "opp_001"


def test_match_result():
    result = MatchResult(
        opportunity_id="opp_001",
        match_score=91.0,
        eligible=True,
    )

    assert result.match_score == 91.0
    assert result.eligible is True


def test_ats_report():
    report = ATSReport(
        ats_score=82.0,
        missing_keywords=["Docker"],
    )

    assert report.ats_score == 82.0


def test_application_package():
    package = ApplicationPackage(
        cover_letter="Test cover letter",
        checklist=["Upload transcript"],
    )

    assert package.cover_letter is not None
    assert "Upload transcript" in package.checklist
def test_resume_optimization_result_defaults():
    result = ResumeOptimizationResult(
        optimized_resume="Optimized resume text"
    )

    assert result.optimized_resume == "Optimized resume text"
    assert result.changed_sections == []
    assert result.added_keywords == []
    assert result.removed_or_weak_content == []
    assert result.recommendations == []