"""
Phase 2a -- Generates 100 synthetic candidate resume PDFs.
Output : backend/data/resumes/candidate_001.pdf ... candidate_100.pdf
Metadata: backend/data/profiles.json  (structured data for Phase 2b indexing)
"""
from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path

from fpdf import FPDF, XPos, YPos

# ── reproducibility ───────────────────────────────────────────────────────────
random.seed(42)

# ── output paths ──────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parents[2] / "data"
RESUME_DIR = BASE_DIR / "resumes"
PROFILES_JSON = BASE_DIR / "profiles.json"

# ── name pools ────────────────────────────────────────────────────────────────
FIRST_NAMES = [
    "James", "Sarah", "Michael", "Emily", "David", "Jessica", "Daniel", "Ashley",
    "Matthew", "Amanda", "Christopher", "Stephanie", "Andrew", "Nicole", "Joshua",
    "Elizabeth", "Ryan", "Megan", "Justin", "Hannah", "Kevin", "Rachel", "Brandon",
    "Lauren", "Tyler", "Samantha", "Adam", "Christina", "Nathan", "Brittany",
    "Raj", "Priya", "Arjun", "Ananya", "Vikram", "Deepa", "Arun", "Neha",
    "Wei", "Mei", "Jun", "Lin", "Yuki", "Hana", "Kenji", "Sakura",
    "Lars", "Anna", "Erik", "Sofia", "Mikael", "Emma", "Nils", "Ingrid",
    "Mohammed", "Fatima", "Omar", "Layla", "Ahmed", "Nadia",
    "Pierre", "Marie", "Jean", "Claire", "Louis", "Julie",
    "Carlos", "Maria", "Diego", "Valentina", "Andres", "Isabella",
    "Luca", "Giulia", "Marco", "Elena", "Alessandro", "Chiara",
    "Olga", "Ivan", "Dmitri", "Natasha", "Pavel", "Katerina",
    "Kwame", "Amara", "Kofi", "Abena", "Yaw", "Ama",
]

LAST_NAMES = [
    "Smith", "Johnson", "Williams", "Brown", "Jones", "Garcia", "Miller", "Davis",
    "Wilson", "Anderson", "Taylor", "Thomas", "Jackson", "White", "Harris",
    "Martin", "Thompson", "Moore", "Young", "Allen", "King", "Wright", "Scott",
    "Patel", "Shah", "Kumar", "Sharma", "Singh", "Gupta", "Mehta", "Nair",
    "Chen", "Wang", "Zhang", "Liu", "Yang", "Tanaka", "Yamamoto", "Suzuki",
    "Larsson", "Johansson", "Eriksson", "Lindgren", "Berg", "Holm",
    "Ibrahim", "Rahman", "Al-Hassan", "Dubois", "Laurent", "Bernard",
    "Rodriguez", "Martinez", "Hernandez", "Lopez", "Ferrari", "Rossi",
    "Mueller", "Schmidt", "Fischer", "Weber", "Kowalski", "Nowak",
    "Petrov", "Ivanova", "Sokolov", "Mensah", "Asante", "Osei",
]

# ── role profiles ─────────────────────────────────────────────────────────────
ROLE_PROFILES: dict[str, dict] = {
    "Frontend Engineer": {
        "primary": ["React", "TypeScript", "JavaScript", "HTML5", "CSS3"],
        "secondary": ["Next.js", "Vue.js", "Redux", "GraphQL", "Webpack", "Jest", "Storybook", "Tailwind CSS"],
        "tools": ["Git", "Figma", "VS Code", "ESLint", "Prettier"],
        "industries": ["fintech", "e-commerce", "SaaS", "media", "healthtech"],
    },
    "Backend Engineer": {
        "primary": ["Python", "Node.js", "PostgreSQL", "REST APIs", "Docker"],
        "secondary": ["FastAPI", "Django", "Redis", "MongoDB", "Kafka", "gRPC", "Kubernetes", "RabbitMQ"],
        "tools": ["Git", "AWS", "Linux", "Postman", "Datadog"],
        "industries": ["fintech", "SaaS", "logistics", "healthtech", "insurance"],
    },
    "Full Stack Engineer": {
        "primary": ["React", "Python", "Node.js", "PostgreSQL", "TypeScript"],
        "secondary": ["FastAPI", "Next.js", "Redis", "Docker", "GraphQL", "AWS", "MongoDB"],
        "tools": ["Git", "VS Code", "Figma", "Linux", "CI/CD"],
        "industries": ["SaaS", "e-commerce", "fintech", "education", "media"],
    },
    "Machine Learning Engineer": {
        "primary": ["Python", "PyTorch", "TensorFlow", "scikit-learn", "SQL"],
        "secondary": ["Hugging Face", "MLflow", "Apache Spark", "Pandas", "NumPy", "ONNX", "Kubeflow"],
        "tools": ["Git", "Jupyter", "Docker", "AWS SageMaker", "Weights and Biases"],
        "industries": ["fintech", "healthtech", "e-commerce", "autonomous vehicles", "media"],
    },
    "Data Scientist": {
        "primary": ["Python", "R", "SQL", "Pandas", "NumPy"],
        "secondary": ["scikit-learn", "XGBoost", "Tableau", "Power BI", "dbt", "Spark", "Statsmodels"],
        "tools": ["Jupyter", "Git", "Databricks", "Snowflake", "Airflow"],
        "industries": ["fintech", "insurance", "healthtech", "retail", "consulting"],
    },
    "Data Engineer": {
        "primary": ["Python", "Apache Spark", "SQL", "Airflow", "dbt"],
        "secondary": ["Snowflake", "BigQuery", "Kafka", "AWS Glue", "Redshift", "Terraform", "Docker"],
        "tools": ["Git", "Linux", "Databricks", "Fivetran", "Great Expectations"],
        "industries": ["fintech", "e-commerce", "logistics", "media", "SaaS"],
    },
    "DevOps Engineer": {
        "primary": ["Kubernetes", "Docker", "Terraform", "AWS", "CI/CD"],
        "secondary": ["Helm", "Ansible", "GCP", "Azure", "Prometheus", "Grafana", "Jenkins", "ArgoCD"],
        "tools": ["Git", "Linux", "Bash", "Python", "GitHub Actions"],
        "industries": ["SaaS", "fintech", "e-commerce", "gaming", "healthtech"],
    },
    "Site Reliability Engineer": {
        "primary": ["Kubernetes", "Prometheus", "Terraform", "Python", "Linux"],
        "secondary": ["Grafana", "AWS", "PagerDuty", "Chaos Engineering", "Istio", "ELK Stack"],
        "tools": ["Git", "Bash", "Datadog", "Ansible", "Docker"],
        "industries": ["SaaS", "fintech", "e-commerce", "media", "gaming"],
    },
    "Mobile Engineer": {
        "primary": ["Swift", "Kotlin", "React Native", "Flutter", "Dart"],
        "secondary": ["SwiftUI", "Jetpack Compose", "Firebase", "CoreData", "GraphQL", "REST APIs"],
        "tools": ["Xcode", "Android Studio", "Git", "Fastlane", "TestFlight"],
        "industries": ["fintech", "healthtech", "e-commerce", "media", "gaming"],
    },
    "Cloud Engineer": {
        "primary": ["AWS", "Terraform", "Kubernetes", "Python", "Docker"],
        "secondary": ["GCP", "Azure", "CloudFormation", "Serverless", "Lambda", "EKS", "S3"],
        "tools": ["Git", "Linux", "Bash", "Ansible", "Vault"],
        "industries": ["SaaS", "fintech", "logistics", "media", "consulting"],
    },
    "Security Engineer": {
        "primary": ["Penetration Testing", "Python", "AWS Security", "SIEM", "IAM"],
        "secondary": ["OWASP", "Burp Suite", "Terraform", "SOC2", "Zero Trust", "SAST/DAST"],
        "tools": ["Git", "Linux", "Wireshark", "Splunk", "CrowdStrike"],
        "industries": ["fintech", "healthtech", "government", "insurance", "SaaS"],
    },
    "QA Engineer": {
        "primary": ["Selenium", "Cypress", "Python", "JavaScript", "REST API Testing"],
        "secondary": ["Playwright", "Jest", "Postman", "JMeter", "Appium", "TestRail", "BDD"],
        "tools": ["Git", "JIRA", "Docker", "CI/CD", "Allure"],
        "industries": ["fintech", "e-commerce", "SaaS", "healthtech", "gaming"],
    },
    "Platform Engineer": {
        "primary": ["Kubernetes", "Python", "Terraform", "Docker", "Linux"],
        "secondary": ["Helm", "Crossplane", "ArgoCD", "Backstage", "Service Mesh", "Vault"],
        "tools": ["Git", "Bash", "Prometheus", "Grafana", "PagerDuty"],
        "industries": ["SaaS", "fintech", "e-commerce", "media", "consulting"],
    },
    "AI Engineer": {
        "primary": ["Python", "LangChain", "OpenAI API", "FastAPI", "Docker"],
        "secondary": ["Pinecone", "ChromaDB", "Hugging Face", "LlamaIndex", "Redis", "PostgreSQL"],
        "tools": ["Git", "Jupyter", "AWS", "Weights and Biases", "MLflow"],
        "industries": ["SaaS", "fintech", "healthtech", "legal tech", "e-commerce"],
    },
    "Embedded Systems Engineer": {
        "primary": ["C", "C++", "RTOS", "ARM", "Linux Kernel"],
        "secondary": ["Python", "CAN Bus", "MQTT", "FreeRTOS", "Rust", "Assembly"],
        "tools": ["Git", "GDB", "JTAG", "Oscilloscope", "Make"],
        "industries": ["automotive", "IoT", "aerospace", "industrial", "healthtech"],
    },
}

SENIORITY = [
    ("Junior",    1,  2),
    ("Mid-Level", 3,  5),
    ("Senior",    6,  9),
    ("Staff",    10, 13),
    ("Principal",14, 18),
]

LOCATIONS = [
    "Berlin, Germany", "London, UK", "Amsterdam, Netherlands",
    "New York, USA", "San Francisco, USA", "Austin, USA", "Seattle, USA",
    "Toronto, Canada", "Vancouver, Canada",
    "Bangalore, India", "Hyderabad, India",
    "Singapore", "Sydney, Australia",
    "Paris, France", "Warsaw, Poland", "Lisbon, Portugal",
    "Dublin, Ireland", "Stockholm, Sweden", "Zurich, Switzerland",
    "Remote",
]

COMPANIES: list[dict] = [
    {"name": "FinFlow Technologies",    "industry": "fintech"},
    {"name": "PaySphere Inc.",          "industry": "fintech"},
    {"name": "TradeLens Corp.",         "industry": "fintech"},
    {"name": "Nexus Health Systems",    "industry": "healthtech"},
    {"name": "MediSync AI",             "industry": "healthtech"},
    {"name": "ShopStream Ltd.",         "industry": "e-commerce"},
    {"name": "CartLogic GmbH",          "industry": "e-commerce"},
    {"name": "Pixel Studios",           "industry": "gaming"},
    {"name": "ArenaX Games",            "industry": "gaming"},
    {"name": "CloudAxis Inc.",          "industry": "SaaS"},
    {"name": "ScaleOps Platform",       "industry": "SaaS"},
    {"name": "DataBridge Analytics",    "industry": "SaaS"},
    {"name": "RapidRoute Logistics",    "industry": "logistics"},
    {"name": "FreightIQ",               "industry": "logistics"},
    {"name": "InsureNow Group",         "industry": "insurance"},
    {"name": "CoverTech AG",            "industry": "insurance"},
    {"name": "Bright Education Labs",   "industry": "education"},
    {"name": "Streamly Media",          "industry": "media"},
    {"name": "ContentFlow Inc.",        "industry": "media"},
    {"name": "AutoDrive Systems",       "industry": "automotive"},
    {"name": "Apex Consulting",         "industry": "consulting"},
    {"name": "TechForge Solutions",     "industry": "consulting"},
    {"name": "Nova Software",           "industry": "SaaS"},
    {"name": "Orbit Technologies",      "industry": "SaaS"},
    {"name": "PulseData Corp.",         "industry": "fintech"},
    {"name": "GreenChain Network",      "industry": "logistics"},
    {"name": "SecureVault Systems",     "industry": "fintech"},
    {"name": "BioTrack Health",         "industry": "healthtech"},
    {"name": "QuantEdge Capital",       "industry": "fintech"},
    {"name": "ClearSky Analytics",      "industry": "SaaS"},
]

UNIVERSITIES = [
    ("MIT",                          "B.Sc. Computer Science"),
    ("Stanford University",          "B.Sc. Computer Science"),
    ("Carnegie Mellon University",   "B.Sc. Software Engineering"),
    ("University of Cambridge",      "B.Eng. Computer Science"),
    ("ETH Zurich",                   "B.Sc. Computer Science"),
    ("TU Berlin",                    "B.Sc. Informatics"),
    ("University of Toronto",        "B.Sc. Computer Science"),
    ("IIT Bombay",                   "B.Tech. Computer Science"),
    ("NUS Singapore",                "B.Comp. Computer Science"),
    ("University of Warsaw",         "M.Sc. Computer Science"),
    ("University of Amsterdam",      "B.Sc. Artificial Intelligence"),
    ("University College London",    "M.Eng. Software Engineering"),
    ("Georgia Tech",                 "B.Sc. Computer Science"),
    ("EPFL Lausanne",                "M.Sc. Data Science"),
    ("University of Sydney",         "B.Sc. Computing"),
]

CERTIFICATIONS: dict[str, list[str]] = {
    "Frontend Engineer":          ["AWS Certified Developer", "Google UX Design Certificate", "Meta Front-End Developer"],
    "Backend Engineer":           ["AWS Certified Solutions Architect", "CKA Kubernetes", "Google Cloud Professional"],
    "Full Stack Engineer":        ["AWS Certified Developer", "MongoDB Certified Developer", "Hashicorp Terraform Associate"],
    "Machine Learning Engineer":  ["AWS Certified ML Specialty", "Google Professional ML Engineer", "Deep Learning Specialization"],
    "Data Scientist":             ["Databricks Certified Associate", "Google Professional Data Engineer", "Tableau Desktop Specialist"],
    "Data Engineer":              ["Databricks Certified Data Engineer", "dbt Analytics Engineering", "AWS Data Analytics Specialty"],
    "DevOps Engineer":            ["CKA Kubernetes", "AWS DevOps Professional", "Hashicorp Terraform Associate"],
    "Site Reliability Engineer":  ["CKA Kubernetes", "Google SRE Foundation", "AWS Advanced Networking"],
    "Mobile Engineer":            ["Google Associate Android Developer", "Apple Swift Certification", "AWS Mobile"],
    "Cloud Engineer":             ["AWS Solutions Architect Professional", "GCP Professional Cloud Architect", "Azure Solutions Architect"],
    "Security Engineer":          ["CISSP", "CEH Certified Ethical Hacker", "AWS Security Specialty"],
    "QA Engineer":                ["ISTQB Advanced Level", "Cypress.io Certified", "AWS Certified Developer"],
    "Platform Engineer":          ["CKA Kubernetes", "Hashicorp Vault Associate", "GitOps Fundamentals"],
    "AI Engineer":                ["AWS Certified ML Specialty", "DeepLearning.AI LLMOps", "Google Cloud GenAI"],
    "Embedded Systems Engineer":  ["ARM Accredited Engineer", "AUTOSAR Certified", "IEC 61508 Functional Safety"],
}

METRICS   = ["latency", "deployment time", "error rate", "infrastructure cost", "page load time", "build time"]
SCALES    = ["50K", "200K", "1M", "5M", "10M", "50M"]
ACTIONS   = [
    "refactoring the data pipeline", "introducing caching layers",
    "migrating to microservices", "optimising database queries",
    "implementing CDN edge caching", "adopting event-driven architecture",
]
STAKEHOLDERS = ["product", "design", "data science", "business intelligence"]
SYSTEMS   = [
    "real-time dashboard", "payment processing pipeline", "recommendation engine",
    "CI/CD platform", "authentication service", "data ingestion pipeline",
    "API gateway", "notification system", "analytics platform", "search service",
]
TECHS = ["Kubernetes", "React", "FastAPI", "Kafka", "Redis", "GraphQL", "Terraform"]

ACHIEVEMENT_TEMPLATES = [
    "Reduced {metric} by {pct}% through {action}",
    "Led migration of {system} to {tech}, improving {metric} by {pct}%",
    "Built {system} from scratch, serving {scale} users",
    "Architected {system} handling {scale} requests per second",
    "Mentored team of {n} engineers, improving sprint velocity by {pct}%",
    "Designed and shipped {system} on time and {pct}% under budget",
    "Owned {system} end-to-end, achieving {pct}% uptime SLA",
    "Collaborated with {stakeholder} team to deliver {system} for {scale} customers",
]


# ── data model ────────────────────────────────────────────────────────────────

@dataclass
class WorkExperience:
    company: str
    industry: str
    title: str
    location: str
    start_year: int
    end_year: int | None
    bullets: list[str]


@dataclass
class Education:
    university: str
    degree: str
    grad_year: int


@dataclass
class CandidateProfile:
    candidate_id: str
    name: str
    title: str
    location: str
    email: str
    phone: str
    linkedin: str
    years_experience: int
    skills: list[str]
    summary: str
    experience: list[WorkExperience]
    education: Education
    certifications: list[str]
    last_updated: str


# ── profile builder ───────────────────────────────────────────────────────────

def _random_bullet() -> str:
    tpl = random.choice(ACHIEVEMENT_TEMPLATES)
    return tpl.format(
        metric=random.choice(METRICS),
        pct=random.choice([15, 20, 25, 30, 35, 40, 50]),
        action=random.choice(ACTIONS),
        system=random.choice(SYSTEMS),
        tech=random.choice(TECHS),
        scale=random.choice(SCALES),
        n=random.randint(2, 8),
        stakeholder=random.choice(STAKEHOLDERS),
    )


def _build_experience(role: str, years: int, current_location: str) -> list[WorkExperience]:
    experiences: list[WorkExperience] = []
    current_year = 2025
    remaining = years

    for i in range(3):
        if remaining <= 0:
            break
        duration = min(random.randint(1, 4), remaining)
        company = random.choice(COMPANIES)
        end = current_year
        start = end - duration

        if remaining >= 10:
            prefix = "Principal "
        elif remaining >= 7:
            prefix = "Senior "
        elif remaining >= 4:
            prefix = ""
        else:
            prefix = "Junior "

        experiences.append(WorkExperience(
            company=company["name"],
            industry=company["industry"],
            title=f"{prefix}{role}",
            location=current_location if i == 0 else random.choice(LOCATIONS),
            start_year=start,
            end_year=None if i == 0 else end,
            bullets=[_random_bullet() for _ in range(random.randint(3, 4))],
        ))
        current_year = start
        remaining -= duration

    return experiences


def _build_summary(name: str, role: str, years: int, skills: list[str], industries: list[str]) -> str:
    first = name.split()[0]
    top_skills = ", ".join(skills[:4])
    industry_str = " and ".join(random.sample(industries, min(2, len(industries))))
    seniority_word = (
        "junior" if years <= 2 else
        "mid-level" if years <= 5 else
        "senior" if years <= 9 else
        "staff-level"
    )
    return (
        f"{first} is a {seniority_word} {role} with {years} years of experience "
        f"building robust software solutions. Proficient in {top_skills}, with a strong "
        f"background in {industry_str} domains. Known for delivering high-quality, "
        f"scalable systems and collaborating effectively across cross-functional teams. "
        f"Passionate about clean code, continuous learning, and engineering excellence."
    )


def build_profile(index: int) -> CandidateProfile:
    role = random.choice(list(ROLE_PROFILES.keys()))
    profile_data = ROLE_PROFILES[role]

    seniority_label, yr_min, yr_max = random.choice(SENIORITY)
    years = random.randint(yr_min, yr_max)
    title = f"{seniority_label} {role}"

    first = random.choice(FIRST_NAMES)
    last  = random.choice(LAST_NAMES)
    name  = f"{first} {last}"
    location = random.choice(LOCATIONS)

    slug    = name.lower().replace(" ", ".").replace("-", "")
    email   = f"{slug}@example.com"
    phone   = f"+{random.randint(1, 99)} {random.randint(100,999)}-{random.randint(100,999)}-{random.randint(1000,9999)}"
    linkedin = f"linkedin.com/in/{slug}"

    skills = (
        profile_data["primary"]
        + random.sample(profile_data["secondary"], k=min(4, len(profile_data["secondary"])))
        + random.sample(profile_data["tools"],     k=min(2, len(profile_data["tools"])))
    )
    random.shuffle(skills)

    industries = profile_data["industries"]
    summary    = _build_summary(name, role, years, skills, industries)
    experience = _build_experience(role, years, location)

    uni, degree = random.choice(UNIVERSITIES)
    grad_year   = 2025 - years - random.randint(0, 2)
    education   = Education(university=uni, degree=degree, grad_year=grad_year)

    certs_pool = CERTIFICATIONS.get(role, [])
    certs      = random.sample(certs_pool, k=min(2, len(certs_pool)))

    days_ago     = random.randint(0, 730)
    last_updated = (date(2025, 6, 5) - timedelta(days=days_ago)).isoformat()

    return CandidateProfile(
        candidate_id=f"candidate_{index:03d}",
        name=name,
        title=title,
        location=location,
        email=email,
        phone=phone,
        linkedin=linkedin,
        years_experience=years,
        skills=skills,
        summary=summary,
        experience=experience,
        education=education,
        certifications=certs,
        last_updated=last_updated,
    )


# ── PDF renderer ──────────────────────────────────────────────────────────────

PRIMARY_COLOR = (30, 58, 95)
ACCENT_COLOR  = (44, 82, 130)
TEXT_COLOR    = (30, 30, 30)
LIGHT_COLOR   = (100, 116, 139)
BG_LIGHT      = (241, 245, 249)

NL   = {"new_x": XPos.LMARGIN, "new_y": YPos.NEXT}   # newline after cell
SAME = {"new_x": XPos.RIGHT,   "new_y": YPos.TOP}    # stay on same line


class ResumePDF(FPDF):
    def __init__(self, profile: CandidateProfile) -> None:
        super().__init__(orientation="P", unit="mm", format="A4")
        self.profile = profile
        self.set_margins(20, 15, 20)
        self.set_auto_page_break(auto=True, margin=15)

    def _set_primary(self) -> None: self.set_text_color(*PRIMARY_COLOR)
    def _set_accent(self)  -> None: self.set_text_color(*ACCENT_COLOR)
    def _set_text(self)    -> None: self.set_text_color(*TEXT_COLOR)
    def _set_light(self)   -> None: self.set_text_color(*LIGHT_COLOR)

    def _section_header(self, title: str) -> None:
        self.ln(4)
        self.set_font("Helvetica", "B", 11)
        self._set_accent()
        self.cell(0, 6, title.upper(), **SAME)
        self.ln(1)
        self.set_draw_color(*ACCENT_COLOR)
        self.set_line_width(0.4)
        y = self.get_y()
        self.line(self.l_margin, y, self.w - self.r_margin, y)
        self.ln(4)
        self._set_text()

    def _bullet(self, text: str) -> None:
        self.set_font("Helvetica", "", 9)
        self._set_text()
        y = self.get_y()
        self.set_xy(self.l_margin + 2, y)
        self.cell(4, 5, "-", **SAME)
        self.set_xy(self.l_margin + 7, y)
        self.multi_cell(self.w - self.l_margin - self.r_margin - 7, 5, text)

    def _render_header(self) -> None:
        p = self.profile
        self.set_fill_color(*PRIMARY_COLOR)
        self.rect(0, 0, self.w, 38, "F")

        self.set_xy(self.l_margin, 7)
        self.set_font("Helvetica", "B", 20)
        self.set_text_color(255, 255, 255)
        self.cell(0, 9, p.name, **NL)

        self.set_x(self.l_margin)
        self.set_font("Helvetica", "", 11)
        self.set_text_color(190, 210, 235)
        self.cell(0, 6, p.title, **NL)

        self.set_xy(self.l_margin, 40)
        self._set_light()
        self.set_font("Helvetica", "", 8.5)
        contact = f"  {p.location}   |   {p.email}   |   {p.phone}   |   {p.linkedin}"
        self.cell(0, 5, contact, **NL)
        self.ln(3)

    def _render_summary(self) -> None:
        self._section_header("Professional Summary")
        self.set_font("Helvetica", "", 9.5)
        self._set_text()
        self.multi_cell(0, 5.5, self.profile.summary)

    def _render_skills(self) -> None:
        self._section_header("Technical Skills")
        skills_text = "  |  ".join(self.profile.skills)
        self.set_fill_color(*BG_LIGHT)
        self.set_font("Helvetica", "", 9)
        self._set_text()
        self.multi_cell(0, 6, skills_text, fill=True)

    def _render_experience_entry(self, exp: WorkExperience) -> None:
        end_str  = "Present" if exp.end_year is None else str(exp.end_year)
        date_str = f"{exp.start_year} - {end_str}"

        self.set_font("Helvetica", "B", 10)
        self._set_primary()
        self.cell(0, 5, exp.title, **NL)

        self.set_font("Helvetica", "I", 9)
        self._set_accent()
        self.cell(100, 5, f"{exp.company}  ({exp.industry})", **SAME)

        self.set_font("Helvetica", "", 9)
        self._set_light()
        self.cell(0, 5, date_str, align="R", **NL)
        self.ln(1)

        for bullet in exp.bullets:
            self._bullet(bullet)
        self.ln(3)

    def _render_education(self) -> None:
        self._section_header("Education")
        edu = self.profile.education
        self.set_font("Helvetica", "B", 10)
        self._set_primary()
        self.cell(0, 5, edu.university, **NL)
        self.set_font("Helvetica", "", 9)
        self._set_text()
        self.cell(0, 5, f"{edu.degree}  |  Graduated {edu.grad_year}", **NL)

    def _render_certifications(self) -> None:
        if not self.profile.certifications:
            return
        self._section_header("Certifications")
        for cert in self.profile.certifications:
            self._bullet(cert)

    def render(self) -> None:
        # ── page 1 ────────────────────────────────────────────────────────────
        self.add_page()
        self._render_header()
        self._render_summary()
        self._render_skills()
        self._section_header("Work Experience")
        for exp in self.profile.experience[:2]:
            self._render_experience_entry(exp)

        # ── page 2 ────────────────────────────────────────────────────────────
        self.add_page()

        self.set_font("Helvetica", "B", 13)
        self._set_primary()
        self.cell(0, 7, self.profile.name, **SAME)
        self.set_font("Helvetica", "", 10)
        self._set_light()
        self.cell(0, 7, f"  -  {self.profile.title} (continued)", **NL)
        self.set_draw_color(*ACCENT_COLOR)
        self.set_line_width(0.3)
        self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
        self.ln(5)

        remaining_exp = self.profile.experience[2:]
        if remaining_exp:
            self._section_header("Work Experience (continued)")
            for exp in remaining_exp:
                self._render_experience_entry(exp)

        self._render_education()
        self._render_certifications()

        # last-updated footer
        self.set_y(-18)
        self.set_font("Helvetica", "I", 7.5)
        self._set_light()
        self.cell(
            0, 5,
            f"Profile last updated: {self.profile.last_updated}   |   ID: {self.profile.candidate_id}",
            align="C",
        )


# ── orchestrator ──────────────────────────────────────────────────────────────

def generate_candidates(count: int = 100) -> list[CandidateProfile]:
    return [build_profile(i + 1) for i in range(count)]


def save_pdfs(profiles: list[CandidateProfile]) -> None:
    RESUME_DIR.mkdir(parents=True, exist_ok=True)
    for i, profile in enumerate(profiles, 1):
        pdf = ResumePDF(profile)
        pdf.render()
        out_path = RESUME_DIR / f"{profile.candidate_id}.pdf"
        pdf.output(str(out_path))
        if i % 10 == 0:
            print(f"  {i}/{len(profiles)} PDFs written...")
    print(f"Saved {len(profiles)} PDFs -> {RESUME_DIR}")


def save_metadata(profiles: list[CandidateProfile]) -> None:
    BASE_DIR.mkdir(parents=True, exist_ok=True)
    data = []
    for p in profiles:
        d = asdict(p)
        data.append(d)
    PROFILES_JSON.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved metadata -> {PROFILES_JSON}")


if __name__ == "__main__":
    print("Generating 100 synthetic candidate profiles...")
    profiles = generate_candidates(100)
    print("Rendering PDFs...")
    save_pdfs(profiles)
    print("Saving metadata JSON...")
    save_metadata(profiles)
    print("\nDone! Check backend/data/resumes/")
