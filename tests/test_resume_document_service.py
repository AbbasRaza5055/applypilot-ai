from pathlib import Path

import pytest

from services.resume_document_service import (
    EmptyResumeError,
    ResumeDocumentGenerationError,
    generate_resume_docx,
    generate_resume_pdf,
)


RESUME_TEXT = """John Doe

AI/ML Engineer

Skills:
Python, Machine Learning

Experience:
AI/ML Intern

Projects:
Resume Screening Assistant
"""


def test_generate_resume_docx(tmp_path):
    output = tmp_path / "resume.docx"

    result = generate_resume_docx(RESUME_TEXT, str(output))

    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0


def test_generate_resume_pdf(tmp_path):
    output = tmp_path / "resume.pdf"

    result = generate_resume_pdf(RESUME_TEXT, str(output))

    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0


@pytest.mark.parametrize("generator", [
    generate_resume_docx,
    generate_resume_pdf,
])
def test_empty_resume_rejected(generator, tmp_path):
    output = tmp_path / "resume"

    with pytest.raises(EmptyResumeError):
        generator("", str(output))


@pytest.mark.parametrize("generator", [
    generate_resume_docx,
    generate_resume_pdf,
])
def test_whitespace_resume_rejected(generator, tmp_path):
    output = tmp_path / "resume"

    with pytest.raises(EmptyResumeError):
        generator("   \n\n  ", str(output))


@pytest.mark.parametrize("generator", [
    generate_resume_docx,
    generate_resume_pdf,
])
def test_invalid_resume_type_rejected(generator, tmp_path):
    output = tmp_path / "resume"

    with pytest.raises(TypeError):
        generator(None, str(output))


@pytest.mark.parametrize("generator", [
    generate_resume_docx,
    generate_resume_pdf,
])
def test_invalid_output_path_type_rejected(generator):
    with pytest.raises(TypeError):
        generator(RESUME_TEXT, None)


def test_docx_creates_parent_directory(tmp_path):
    output = tmp_path / "nested" / "folder" / "resume.docx"

    result = generate_resume_docx(RESUME_TEXT, str(output))

    assert result == str(output)
    assert output.exists()


def test_pdf_creates_parent_directory(tmp_path):
    output = tmp_path / "nested" / "folder" / "resume.pdf"

    result = generate_resume_pdf(RESUME_TEXT, str(output))

    assert result == str(output)
    assert output.exists()


def test_pdf_blank_lines_force_page_boundary(tmp_path):
    # ~60 blank lines exceed a single letter-page (792pt height, 50pt margins,
    # 15pt line_height → ~46 lines per page), guaranteeing the blank-line
    # pagination path is exercised.
    resume = "John Doe\n" + "\n" * 60 + "Skills: Python"
    output = tmp_path / "resume_blank_lines.pdf"

    result = generate_resume_pdf(resume, str(output))

    assert result == str(output)
    assert output.exists()
    assert output.stat().st_size > 0
