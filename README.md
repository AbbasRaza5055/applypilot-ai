# ApplyPilot AI

> **One Profile. Every Opportunity. Zero Missed Steps.**

ApplyPilot AI is an AI-powered career application assistant that helps students and early-career professionals move from a raw resume to a more prepared job or opportunity application through a single workflow.

The platform transforms an uploaded CV into a structured candidate profile, discovers relevant opportunities, evaluates fit, analyzes ATS readiness, optimizes the resume, and prepares application materials.

---

## 🌟 Live Demo

**Streamlit App:**  
https://applypilotai.streamlit.app/

**GitHub Repository:**  
https://github.com/AbbasRaza5055/applypilot-ai

---

## 🎯 Problem

Finding and applying for opportunities is fragmented and time-consuming.

A candidate typically has to:

- Search multiple platforms for jobs, internships, scholarships, hackathons, and fellowships.
- Manually check whether an opportunity matches their profile.
- Read job requirements and compare them with their skills.
- Understand why a resume may perform poorly in an ATS.
- Rewrite a resume for different opportunities.
- Write cover letters, SOPs, and application answers.
- Keep track of required documents and application steps.

This creates a repetitive workflow with a lot of manual effort and missed opportunities.

---

## 💡 Solution

**ApplyPilot AI** brings these steps into one guided workflow:

```text
Upload Resume
      ↓
Profile Intelligence
      ↓
Opportunity Discovery
      ↓
Opportunity Matching
      ↓
ATS Analysis
      ↓
Resume Optimization
      ↓
Application Package
```

The goal is to help candidates spend less time on repetitive application preparation and more time on opportunities that actually fit their profile.

---

## 🚀 Core Workflow

### 1. Profile Intelligence

The user uploads a PDF or DOCX resume.

ApplyPilot extracts the resume text and uses AI to create a structured candidate profile containing information such as:

- Name
- Email
- Phone
- Location
- Education
- Skills
- Experience
- Projects
- Certifications
- Interests
- Years of experience

---

### 2. Opportunity Discovery

ApplyPilot searches for relevant opportunities based on the candidate profile.

Supported opportunity types include:

- Jobs
- Internships
- Scholarships
- Hackathons
- Fellowships

Discovered opportunities are normalized into a common internal structure so they can be processed consistently by the rest of the application pipeline.

---

### 3. Opportunity Matching

The matching engine evaluates how well each opportunity fits the candidate.

The system considers factors such as:

- Skills
- Education
- Experience
- Requirements
- Location and other relevant constraints

Each opportunity receives a match score together with:

- Eligibility
- Strengths
- Gaps
- Reasons for the match

---

### 4. ATS Analysis

ApplyPilot analyzes the candidate's resume against a selected opportunity.

The ATS analysis produces:

- ATS score
- Strengths
- Issues
- Matched keywords
- Missing keywords
- Recommendations

This helps the candidate understand how well the current resume aligns with the opportunity.

---

### 5. Resume Optimization

The resume can then be optimized for the selected opportunity.

The optimization workflow focuses on:

- Improving relevant resume sections
- Adding useful opportunity-related keywords
- Removing or improving weak content
- Providing recommendations

The optimizer is designed to remain grounded in the candidate's existing information rather than inventing new experience or qualifications.

---

### 6. Application Package

After optimization, ApplyPilot prepares an application package that can include:

- Cover letter
- Statement of purpose (SOP)
- Application answers
- Required documents
- Application checklist

The system also supports generating optimized resume documents for download.

---

## 🧠 Architecture

ApplyPilot AI follows a modular architecture where the Streamlit interface orchestrates specialized backend agents and services.

```text
                    ┌─────────────────────┐
                    │    Streamlit UI     │
                    │       app.py        │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Candidate Profile   │
                    │   Profile Agent     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Opportunity         │
                    │ Discovery Agent     │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Matching Agent      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ ATS Agent           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Resume Optimizer    │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Application Agent   │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Resume Documents    │
                    │     DOCX / PDF      │
                    └─────────────────────┘
```

---

## 🏗️ Project Structure

```text
applypilot-ai/
│
├── app.py
├── README.md
├── requirements.txt
├── pytest.ini
├── .gitignore
├── .env.example
│
├── agents/
│   ├── profile_agent.py
│   ├── discovery_agent.py
│   ├── matching_agent.py
│   ├── ats_agent.py
│   ├── resume_optimizer.py
│   └── application_agent.py
│
├── models/
│   ├── profile.py
│   ├── opportunity.py
│   ├── matching.py
│   ├── ats.py
│   ├── resume_optimization.py
│   └── application.py
│
├── services/
│   ├── document_service.py
│   └── resume_document_service.py
│
├── tests/
│   ├── test_profile_agent.py
│   ├── test_discovery_agent.py
│   ├── test_matching_agent.py
│   ├── test_ats_agent.py
│   ├── test_resume_optimizer.py
│   ├── test_application_agent.py
│   ├── test_resume_document_service.py
│   └── test_models.py
│
└── data/
```

---

## 🛠️ Tech Stack

### Frontend / UI

- Streamlit

### Programming Language

- Python

### AI / LLM

- Google Gemini
- Groq

### Data Validation

- Pydantic

### Document Processing

- PyPDF
- python-docx
- ReportLab

### Testing

- pytest

### Development

- Git
- GitHub
- VS Code

---

## 📄 Supported Resume Formats

ApplyPilot currently supports:

- PDF
- DOCX

The resume text is extracted before the AI profile analysis stage.

---

## ⚙️ Local Setup

### 1. Clone the repository

```bash
git clone https://github.com/AbbasRaza5055/applypilot-ai.git
```

### 2. Enter the project directory

```bash
cd applypilot-ai
```

### 3. Create a virtual environment

Windows:

```bash
python -m venv .venv
```

### 4. Activate the virtual environment

#### Windows PowerShell

```powershell
.\.venv\Scripts\Activate.ps1
```

#### Windows CMD

```cmd
.venv\Scripts\activate
```

### 5. Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔐 Environment Variables

Create a local `.env` file in the project root.

Example:

```env
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-3.6-flash

GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-120b

ADZUNA_APP_ID=your_adzuna_app_id_here
ADZUNA_APP_KEY=your_adzuna_app_key_here
ADZUNA_COUNTRY_CODE=gb
```

> **Never commit the real `.env` file or API keys to GitHub.**

The repository includes `.env.example` as a safe configuration template.

---

## ▶️ Run the Application

Start Streamlit:

```bash
streamlit run app.py
```

Then open:

```text
http://localhost:8501
```

---

## 🧪 Running Tests

Run the complete test suite:

```bash
python -m pytest
```

### Current validation

```text
145 tests passed
```

The automated test suite covers:

- Candidate profile extraction
- Opportunity discovery
- Opportunity matching
- ATS analysis
- Resume optimization
- Application package generation
- Resume document generation
- Data model validation

---

## 🔍 Testing Strategy

The project uses unit tests and mocked external API interactions to validate individual agents and services without depending entirely on live API responses.

The tests cover areas such as:

- Valid and invalid inputs
- API configuration errors
- Structured response validation
- URL validation
- Duplicate opportunity removal
- Stable opportunity IDs
- Match result validation
- ATS report validation
- Resume optimization safeguards
- Application package validation
- PDF/DOCX generation

---

## 📦 Generated Resume Documents

ApplyPilot includes deterministic document generation utilities for optimized resumes.

Supported output formats:

- DOCX
- PDF

This separates AI-generated content from the deterministic document-generation layer.

---

## 🔒 Reliability & Safety Principles

### No fabricated candidate information

Resume optimization and application generation should remain grounded in information supplied by the candidate.

### Structured outputs

AI-generated data is validated against Pydantic models before being used by downstream components.

### Modular agents

Each major task has its own agent/module so that individual components can be tested and improved independently.

### API keys remain outside source control

Sensitive credentials are loaded through environment variables and deployment secrets.

---

## 🌐 Live Deployment

The current production deployment is available at:

**https://applypilotai.streamlit.app/**

---

## 📌 Current Scope

ApplyPilot currently focuses on assisting candidates through:

```text
Discover
   ↓
Match
   ↓
Analyze
   ↓
Improve
   ↓
Apply
```

The system prepares application materials and documents but does not automatically submit applications to external websites on behalf of the user.

---

## 🚧 Future Improvements

Potential future improvements include:

- More opportunity sources
- Better location-aware opportunity discovery
- Additional job and scholarship providers
- Improved freshness verification
- Opportunity tracking and reminders
- Application status tracking
- Personalized application analytics
- Browser-assisted application workflows
- Better ranking and recommendation models
- More resume templates and document formats

---

## 👥 Team

### HackForge

A 6-member hackathon team building ApplyPilot AI for the POG Angel Cohort 22 hackathon.

Team members:

- Abbas Raza - Team Lead
- Raja Salman
- Alyan Khan
- Umaima Rizwan
- Sheikh M Muneer
- Abdur Rehman

> Replace the placeholders above with the final team member names and roles before submission.

---

## 🏆 Hackathon Project

**Project:** ApplyPilot AI  
**Team:** HackForge  
**Cohort:** POG Angel Cohort 22

---

## 🔗 Links

### Live Application

https://applypilotai.streamlit.app/

### GitHub Repository

https://github.com/AbbasRaza5055/applypilot-ai

---

## 📜 License

This project is developed as a hackathon project.
