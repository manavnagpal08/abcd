import streamlit as st
import pdfplumber
import pandas as pd
import re
import os
import sklearn
import joblib
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from sentence_transformers import SentenceTransformer
import nltk
import collections
from sklearn.metrics.pairwise import cosine_similarity
import urllib.parse # For encoding mailto links

# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event

# For Generative AI (Google Gemini Pro) - COMMENTED OUT AS PER USER REQUEST
# import google.generativeai as genai

# --- Configure Google Gemini API Key --- - COMMENTED OUT AS PER USER REQUEST
# Store your API key securely in Streamlit Secrets.
# Create a .streamlit/secrets.toml file in your app's directory:
# GOOGLE_API_KEY="YOUR_GEMINI_API_KEY"
# try:
#     genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
# except AttributeError:
#     st.error("🚨 Google API Key not found in Streamlit Secrets. Please add it to your .streamlit/secrets.toml file.")
#     st.stop() # Stop the app if API key is missing

# Download NLTK stopwords data if not already downloaded
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')
    log_system_event("INFO", "NLTK_DOWNLOAD", {"resource": "stopwords"})


# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        log_system_event("INFO", "ML_MODEL_LOADED", {"model_name": "all-MiniLM-L6-v2", "ml_model_file": "ml_screening_model.pkl"})
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        log_system_event("ERROR", "ML_MODEL_LOAD_FAILED", {"error": str(e)})
        return None, None

model, ml_model = load_ml_model()

# --- Stop Words List (Using NLTK) ---
NLTK_STOP_WORDS = set(nltk.corpus.stopwords.words('english'))
CUSTOM_STOP_WORDS = set([
    "work", "experience", "years", "year", "months", "month", "day", "days", "project", "projects",
    "team", "teams", "developed", "managed", "led", "created", "implemented", "designed",
    "responsible", "proficient", "knowledge", "ability", "strong", "proven", "demonstrated",
    "solution", "solutions", "system", "systems", "platform", "platforms", "framework", "frameworks",
    "database", "databases", "server", "servers", "cloud", "computing", "machine", "learning",
    "artificial", "intelligence", "api", "apis", "rest", "graphql", "agile", "scrum", "kanban",
    "devops", "ci", "cd", "testing", "qa",
    "security", "network", "networking", "virtualization",
    "containerization", "docker", "kubernetes", "git", "github", "gitlab", "bitbucket", "jira",
    "confluence", "slack", "microsoft", "google", "amazon", "azure", "oracle", "sap", "crm", "erp",
    "salesforce", "servicenow", "tableau", "powerbi", "qlikview", "excel", "word", "powerpoint",
    "outlook", "visio", "html", "css", "js", "web", "data", "science", "analytics", "engineer",
    "software", "developer", "analyst", "business", "management", "reporting", "analysis", "tools",
    "python", "java", "javascript", "c++", "c#", "php", "ruby", "go", "swift", "kotlin", "r",
    "sql", "nosql", "linux", "unix", "windows", "macos", "ios", "android", "mobile", "desktop",
    "application", "applications", "frontend", "backend", "fullstack", "ui", "ux", "design",
    "architecture", "architect", "engineering", "scientist", "specialist", "consultant",
    "associate", "senior", "junior", "lead", "principal", "director", "manager", "head", "chief",
    "officer", "president", "vice", "executive", "ceo", "cto", "cfo", "coo", "hr", "human",
    "resources", "recruitment", "talent", "acquisition", "onboarding", "training", "development",
    "performance", "compensation", "benefits", "payroll", "compliance", "legal", "finance",
    "accounting", "auditing", "tax", "budgeting", "forecasting", "investments", "marketing",
    "sales", "customer", "service", "support", "operations", "supply", "chain", "logistics",
    "procurement", "manufacturing", "production", "quality", "assurance", "control", "research",
    "innovation", "product", "program", "portfolio", "governance", "risk", "communication",
    "presentation", "negotiation", "problem", "solving", "critical", "thinking", "analytical",
    "creativity", "adaptability", "flexibility", "teamwork", "collaboration", "interpersonal",
    "organizational", "time", "multitasking", "detail", "oriented", "independent", "proactive",
    "self", "starter", "results", "driven", "client", "facing", "stakeholder", "engagement",
    "vendor", "budget", "cost", "reduction", "process", "improvement", "standardization",
    "optimization", "automation", "digital", "transformation", "change", "methodologies",
    "industry", "regulations", "regulatory", "documentation", "technical", "writing",
    "dashboards", "visualizations", "workshops", "feedback", "reviews", "appraisals",
    "offboarding", "employee", "relations", "diversity", "inclusion", "equity", "belonging",
    "corporate", "social", "responsibility", "csr", "sustainability", "environmental", "esg",
    "ethics", "integrity", "professionalism", "confidentiality", "discretion", "accuracy",
    "precision", "efficiency", "effectiveness", "scalability", "robustness", "reliability",
    "vulnerability", "assessment", "penetration", "incident", "response", "disaster",
    "recovery", "continuity", "bcp", "drp", "gdpr", "hipaa", "soc2", "iso", "nist", "pci",
    "dss", "ccpa", "privacy", "protection", "grc", "cybersecurity", "information", "infosec",
    "threat", "intelligence", "soc", "event", "siem", "identity", "access", "iam", "privileged",
    "pam", "multi", "factor", "authentication", "mfa", "single", "sign", "on", "sso",
    "encryption", "decryption", "firewall", "ids", "ips", "vpn", "endpoint", "antivirus",
    "malware", "detection", "forensics", "handling", "assessments", "policies", "procedures",
    "guidelines", "mitre", "att&ck", "modeling", "secure", "lifecycle", "sdlc", "awareness",
    "phishing", "vishing", "smishing", "ransomware", "spyware", "adware", "rootkits",
    "botnets", "trojans", "viruses", "worms", "zero", "day", "exploits", "patches", "patching",
    "updates", "upgrades", "configuration", "ticketing", "crm", "erp", "scm", "hcm", "financial",
    "accounting", "bi", "warehousing", "etl", "extract", "transform", "load", "lineage",
    "master", "mdm", "lakes", "marts", "big", "hadoop", "spark", "kafka", "flink", "mongodb",
    "cassandra", "redis", "elasticsearch", "relational", "mysql", "postgresql", "db2",
    "teradata", "snowflake", "redshift", "synapse", "bigquery", "aurora", "dynamodb",
    "documentdb", "cosmosdb", "graph", "neo4j", "graphdb", "timeseries", "influxdb",
    "timescaledb", "columnar", "vertica", "clickhouse", "vector", "pinecone", "weaviate",
    "milvus", "qdrant", "chroma", "faiss", "annoy", "hnswlib", "scikit", "learn", "tensorflow",
    "pytorch", "keras", "xgboost", "lightgbm", "catboost", "statsmodels", "numpy", "pandas",
    "matplotlib", "seaborn", "plotly", "bokeh", "dash", "flask", "django", "fastapi", "spring",
    "boot", ".net", "core", "node.js", "express.js", "react", "angular", "vue.js", "svelte",
    "jquery", "bootstrap", "tailwind", "sass", "less", "webpack", "babel", "npm", "yarn",
    "ansible", "terraform", "jenkins", "gitlab", "github", "actions", "codebuild", "codepipeline",
    "codedeploy", "build", "deploy", "run", "lambda", "functions", "serverless", "microservices",
    "gateway", "mesh", "istio", "linkerd", "grpc", "restful", "soap", "message", "queues",
    "rabbitmq", "activemq", "bus", "sqs", "sns", "pubsub", "version", "control", "svn",
    "mercurial", "trello", "asana", "monday.com", "smartsheet", "project", "primavera",
    "zendesk", "freshdesk", "itil", "cobit", "prince2", "pmp", "master", "owner", "lean",
    "six", "sigma", "black", "belt", "green", "yellow", "qms", "9001", "27001", "14001",
    "ohsas", "18001", "sa", "8000", "cmii", "cmi", "cism", "cissp", "ceh", "comptia",
    "security+", "network+", "a+", "linux+", "ccna", "ccnp", "ccie", "certified", "solutions",
    "architect", "developer", "sysops", "administrator", "specialty", "professional", "azure",
    "az-900", "az-104", "az-204", "az-303", "az-304", "az-400", "az-500", "az-700", "az-800",
    "az-801", "dp-900", "dp-100", "dp-203", "ai-900", "ai-102", "da-100", "pl-900", "pl-100",
    "pl-200", "pl-300", "pl-400", "pl-500", "ms-900", "ms-100", "ms-101", "ms-203", "ms-500",
    "ms-700", "ms-720", "ms-740", "ms-600", "sc-900", "sc-200", "sc-300", "sc-400", "md-100",
    "md-101", "mb-200", "mb-210", "mb-220", "mb-230", "mb-240", "mb-260", "mb-300", "mb-310",
    "mb-320", "mb-330", "mb-340", "mb-400", "mb-500", "mb-600", "mb-700", "mb-800", "mb-910",
    "mb-920", "gcp-ace", "gcp-pca", "gcp-pde", "gcp-pse", "gcp-pml", "gcp-psa", "gcp-pcd",
    "gcp-pcn", "gcp-psd", "gcp-pda", "gcp-pci", "gcp-pws", "gcp-pwa", "gcp-pme", "gcp-pms",
    "gcp-pmd", "gcp-pma", "gcp-pmc", "gcp-pmg", "cisco", "juniper", "red", "hat", "rhcsa",
    "rhce", "vmware", "vcpa", "vcpd", "vcpi", "vcpe", "vcpx", "citrix", "cc-v", "cc-p",
    "cc-e", "cc-m", "cc-s", "cc-x", "palo", "alto", "pcnsa", "pcnse", "fortinet", "fcsa",
    "fcsp", "fcc", "fcnsp", "fct", "fcp", "fcs", "fce", "fcn", "fcnp", "fcnse"
])
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- MASTER SKILLS LIST ---
# Paste your comprehensive list of skills here.
# These skills will be used to filter words for the word cloud and
# to identify 'Matched Keywords' and 'Missing Skills'.
MASTER_SKILLS = set([
        # Product & Project Management
    "Product Strategy", "Roadmap Development", "Agile Methodologies", "Scrum", "Kanban", "Jira", "Trello",
    "Feature Prioritization", "OKRs", "KPIs", "Stakeholder Management", "A/B Testing", "User Stories", "Epics",
    "Product Lifecycle", "Sprint Planning", "Project Charter", "Gantt Charts", "MVP", "Backlog Grooming",
    "Risk Management", "Change Management", "Program Management", "Portfolio Management", "PMP", "CSM",

    # Software Development & Engineering
    "Python", "Java", "JavaScript", "C++", "C#", "Go", "Ruby", "PHP", "Swift", "Kotlin", "TypeScript",
    "HTML5", "CSS3", "React", "Angular", "Vue.js", "Node.js", "Django", "Flask", "Spring Boot", "Express.js",
    "Git", "GitHub", "GitLab", "Bitbucket", "REST APIs", "GraphQL", "Microservices", "System Design",
    "Unit Testing", "Integration Testing", "End-to-End Testing", "Test Automation", "CI/CD", "Docker", "Kubernetes",
    "Serverless", "AWS Lambda", "Azure Functions", "Google Cloud Functions", "WebSockets", "Kafka", "RabbitMQ",
    "Redis", "SQL", "NoSQL", "PostgreSQL", "MySQL", "MongoDB", "Cassandra", "Elasticsearch", "Neo4j",
    "Data Structures", "Algorithms", "Object-Oriented Programming", "Functional Programming", "Bash Scripting",
    "Shell Scripting", "DevOps", "DevSecOps", "SRE", "CloudFormation", "Terraform", "Ansible", "Puppet", "Chef",
    "Jenkins", "CircleCI", "GitHub Actions", "Azure DevOps", "Jira", "Confluence", "Swagger", "OpenAPI",

    # Data Science & AI/ML
    "Machine Learning", "Deep Learning", "Natural Language Processing", "Computer Vision", "Reinforcement Learning",
    "Scikit-learn", "TensorFlow", "PyTorch", "Keras", "XGBoost", "LightGBM", "Data Cleaning", "Feature Engineering",
    "Model Evaluation", "Statistical Modeling", "Time Series Analysis", "Predictive Modeling", "Clustering",
    "Classification", "Regression", "Neural Networks", "Convolutional Networks", "Recurrent Networks",
    "Transformers", "LLMs", "Prompt Engineering", "Generative AI", "MLOps", "Data Munging", "A/B Testing",
    "Experiment Design", "Hypothesis Testing", "Bayesian Statistics", "Causal Inference", "Graph Neural Networks",

    # Data Analytics & BI
    "SQL", "Python (Pandas, NumPy)", "R", "Excel (Advanced)", "Tableau", "Power BI", "Looker", "Qlik Sense",
    "Google Data Studio", "Dax", "M Query", "ETL", "ELT", "Data Warehousing", "Data Lake", "Data Modeling",
    "Business Intelligence", "Data Visualization", "Dashboarding", "Report Generation", "Google Analytics",
    "BigQuery", "Snowflake", "Redshift", "Data Governance", "Data Quality", "Statistical Analysis",
    "Requirements Gathering", "Data Storytelling",

    # Cloud & Infrastructure
    "AWS", "Azure", "Google Cloud Platform", "GCP", "Cloud Architecture", "Hybrid Cloud", "Multi-Cloud",
    "Virtualization", "VMware", "Hyper-V", "Linux Administration", "Windows Server", "Networking", "TCP/IP",
    "DNS", "VPN", "Firewalls", "Load Balancing", "CDN", "Monitoring", "Logging", "Alerting", "Prometheus",
    "Grafana", "Splunk", "ELK Stack", "Cloud Security", "IAM", "VPC", "Storage (S3, Blob, GCS)", "Databases (RDS, Azure SQL)",
    "Container Orchestration", "Infrastructure as Code", "IaC",

    # UI/UX & Design
    "Figma", "Adobe XD", "Sketch", "Photoshop", "Illustrator", "InDesign", "User Research", "Usability Testing",
    "Wireframing", "Prototyping", "UI Design", "UX Design", "Interaction Design", "Information Architecture",
    "Design Systems", "Accessibility", "Responsive Design", "User Flows", "Journey Mapping", "Design Thinking",
    "Visual Design", "Motion Graphics",

    # Marketing & Sales
    "Digital Marketing", "SEO", "SEM", "Content Marketing", "Email Marketing", "Social Media Marketing",
    "Google Ads", "Facebook Ads", "LinkedIn Ads", "Marketing Automation", "HubSpot", "Salesforce Marketing Cloud",
    "CRM", "Lead Generation", "Sales Strategy", "Negotiation", "Account Management", "Market Research",
    "Campaign Management", "Conversion Rate Optimization", "CRO", "Brand Management", "Public Relations",
    "Copywriting", "Content Creation", "Analytics (Google Analytics, SEMrush, Ahrefs)",

    # Finance & Accounting
    "Financial Modeling", "Valuation", "Financial Reporting", "GAAP", "IFRS", "Budgeting", "Forecasting",
    "Variance Analysis", "Auditing", "Taxation", "Accounts Payable", "Accounts Receivable", "Payroll",
    "QuickBooks", "SAP FICO", "Oracle Financials", "Cost Accounting", "Management Accounting", "Treasury Management",
    "Investment Analysis", "Risk Analysis", "Compliance (SOX, AML)",

    # Human Resources (HR)
    "Talent Acquisition", "Recruitment", "Onboarding", "Employee Relations", "HRIS (Workday, SuccessFactors)",
    "Compensation & Benefits", "Performance Management", "Workforce Planning", "HR Policies", "Labor Law",
    "Training & Development", "Diversity & Inclusion", "Conflict Resolution", "Employee Engagement",

    # Customer Service & Support
    "Customer Relationship Management", "CRM", "Zendesk", "ServiceNow", "Intercom", "Live Chat", "Ticketing Systems",
    "Issue Resolution", "Technical Support", "Customer Success", "Client Retention", "Communication Skills",

    # General Business & Soft Skills (often paired with technical skills)
    "Strategic Planning", "Business Development", "Vendor Management", "Process Improvement", "Operations Management",
    "Project Coordination", "Public Speaking", "Presentation Skills", "Cross-functional Collaboration",
    "Problem Solving", "Critical Thinking", "Analytical Skills", "Adaptability", "Time Management",
    "Organizational Skills", "Attention to Detail", "Leadership", "Mentorship", "Team Leadership",
    "Decision Making", "Negotiation", "Client Management", "Stakeholder Communication", "Active Listening",
    "Creativity", "Innovation", "Research", "Data Analysis", "Report Writing", "Documentation",
    "Microsoft Office Suite", "Google Workspace", "Slack", "Zoom", "Confluence", "SharePoint",
    "Cybersecurity", "Information Security", "Risk Assessment", "Compliance", "GDPR", "HIPAA", "ISO 27001",
    "Penetration Testing", "Vulnerability Management", "Incident Response", "Security Audits", "Forensics",
    "Threat Intelligence", "SIEM", "Firewall Management", "Endpoint Security", "Identity and Access Management",
    "IAM", "Cryptography", "Network Security", "Application Security", "Cloud Security",

    # Specific Certifications/Tools often treated as skills
    "PMP", "CSM", "AWS Certified", "Azure Certified", "GCP Certified", "CCNA", "CISSP", "CISM", "CompTIA Security+",
    "ITIL", "Lean Six Sigma", "CFA", "CPA", "SHRM-CP", "PHR", "CEH", "OSCP", "Splunk", "ServiceNow", "Salesforce",
    "Workday", "SAP", "Oracle", "Microsoft Dynamics", "NetSuite", "Adobe Creative Suite", "Canva", "Mailchimp",
    "Hootsuite", "Buffer", "SEMrush", "Ahrefs", "Moz", "Screaming Frog", "JMeter", "Postman", "SoapUI",
    "Git", "SVN", "Perforce", "Confluence", "Jira", "Asana", "Trello", "Monday.com", "Miro", "Lucidchart",
    "Visio", "MS Project", "Primavera", "AutoCAD", "SolidWorks", "MATLAB", "LabVIEW", "Simulink", "ANSYS",
    "CATIA", "NX", "Revit", "ArcGIS", "QGIS", "OpenCV", "NLTK", "SpaCy", "Gensim", "Hugging Face Transformers",
    "Docker Compose", "Helm", "Ansible Tower", "SaltStack", "Chef InSpec", "Terraform Cloud", "Vault",
    "Consul", "Nomad", "Prometheus", "Grafana", "Alertmanager", "Loki", "Tempo", "Jaeger", "Zipkin",
    "Fluentd", "Logstash", "Kibana", "Grafana Loki", "Datadog", "New Relic", "AppDynamics", "Dynatrace",
    "Nagios", "Zabbix", "Icinga", "PRTG", "SolarWinds", "Wireshark", "Nmap", "Metasploit", "Burp Suite",
    "OWASP ZAP", "Nessus", "Qualys", "Rapid7", "Tenable", "CrowdStrike", "SentinelOne", "Palo Alto Networks",
    "Fortinet", "Cisco Umbrella", "Okta", "Auth0", "Keycloak", "Ping Identity", "Active Directory",
    "LDAP", "OAuth", "JWT", "OpenID Connect", "SAML", "MFA", "SSO", "PKI", "TLS/SSL", "VPN", "IDS/IPS",
    "DLP", "CASB", "SOAR", "XDR", "EDR", "MDR", "GRC", "GDPR Compliance", "HIPAA Compliance", "PCI DSS Compliance",
    "ISO 27001 Compliance", "NIST Framework", "COBIT", "ITIL Framework", "Scrum Master", "Product Owner",
    "Agile Coach", "Release Management", "Change Control", "Configuration Management", "Asset Management",
    "Service Desk", "Incident Management", "Problem Management", "Change Management", "Release Management",
    "Service Level Agreements", "SLAs", "Operational Level Agreements", "OLAs", "Underpinning Contracts", "UCs",
    "Knowledge Management", "Continual Service Improvement", "CSI", "Service Catalog", "Service Portfolio",
    "Relationship Management", "Supplier Management", "Financial Management for IT Services",
    "Demand Management", "Capacity Management", "Availability Management", "Information Security Management",
    "Supplier Relationship Management", "Contract Management", "Procurement Management", "Quality Management",
    "Test Management", "Defect Management", "Requirements Management", "Scope Management", "Time Management",
    "Cost Management", "Quality Management", "Resource Management", "Communications Management",
    "Risk Management", "Procurement Management", "Stakeholder Management", "Integration Management",
    "Project Charter", "Project Plan", "Work Breakdown Structure", "WBS", "Gantt Chart", "Critical Path Method",
    "CPM", "Earned Value Management", "EVM", "PERT", "CPM", "Crashing", "Fast Tracking", "Resource Leveling",
    "Resource Smoothing", "Agile Planning", "Scrum Planning", "Kanban Planning", "Sprint Backlog",
    "Product Backlog", "User Story Mapping", "Relative Sizing", "Planning Poker", "Velocity", "Burndown Chart",
    "Burnup Chart", "Cumulative Flow Diagram", "CFD", "Value Stream Mapping", "VSM", "Lean Principles",
    "Six Sigma", "Kaizen", "Kanban", "Total Quality Management", "TQM", "Statistical Process Control", "SPC",
    "Control Charts", "Pareto Analysis", "Fishbone Diagram", "5 Whys", "FMEA", "Root Cause Analysis", "RCA",
    "Corrective Actions", "Preventive Actions", "CAPA", "Non-conformance Management", "Audit Management",
    "Document Control", "Record Keeping", "Training Management", "Calibration Management", "Supplier Quality Management",
    "Customer Satisfaction Measurement", "Net Promoter Score", "NPS", "Customer Effort Score", "CES",
    "Customer Satisfaction Score", "CSAT", "Voice of Customer", "VOC", "Complaint Handling", "Warranty Management",
    "Returns Management", "Service Contracts", "Service Agreements", "Maintenance Management", "Field Service Management",
    "Asset Management", "Enterprise Asset Management", "EAM", "Computerized Maintenance Management System", "CMMS",
    "Geographic Information Systems", "GIS", "GPS", "Remote Sensing", "Image Processing", "CAD", "CAM", "CAE",
    "FEA", "CFD", "PLM", "PDM", "ERP", "CRM", "SCM", "HRIS", "BI", "Analytics", "Data Science", "Machine Learning",
    "Deep Learning", "NLP", "Computer Vision", "AI", "Robotics", "Automation", "IoT", "Blockchain", "Cybersecurity",
    "Cloud Computing", "Big Data", "Data Warehousing", "ETL", "Data Modeling", "Data Governance", "Data Quality",
    "Data Migration", "Data Integration", "Data Virtualization", "Data Lakehouse", "Data Mesh", "Data Fabric",
    "Data Catalog", "Data Lineage", "Metadata Management", "Master Data Management", "MDM",
    "Customer Data Platform", "CDP", "Digital Twin", "Augmented Reality", "AR", "Virtual Reality", "VR",
    "Mixed Reality", "MR", "Extended Reality", "XR", "Game Development", "Unity", "Unreal Engine", "C# (Unity)",
    "C++ (Unreal Engine)", "Game Design", "Level Design", "Character Design", "Environment Design",
    "Animation (Game)", "Rigging", "Texturing", "Shading", "Lighting", "Rendering", "Game Physics",
    "Game AI", "Multiplayer Networking", "Game Monetization", "Game Analytics", "Playtesting",
    "Game Publishing", "Streaming (Gaming)", "Community Management (Gaming)",
    "Game Art", "Game Audio", "Sound Design (Game)", "Music Composition (Game)", "Voice Acting (Game)",
    "Narrative Design", "Storytelling (Game)", "Dialogue Writing", "World Building", "Lore Creation",
    "Game Scripting", "Modding", "Game Engine Development", "Graphics Programming", "Physics Programming",
    "AI Programming (Game)", "Network Programming (Game)", "Tools Programming (Game)", "UI Programming (Game)",
    "Shader Development", "VFX (Game)", "Technical Art", "Technical Animation", "Technical Design",
    "Build Engineering (Game)", "Release Engineering (Game)", "Live Operations (Game)", "Game Balancing",
    "Economy Design (Game)", "Progression Systems (Game)", "Retention Strategies (Game)", "Monetization Strategies (Game)",
    "User Acquisition (Game)", "Marketing (Game)", "PR (Game)", "Community Management (Game)",
    "Customer Support (Game)", "Localization (Game)", "Quality Assurance (Game)", "Game Testing",
    "Compliance (Game)", "Legal (Game)", "Finance (Game)", "HR (Game)", "Business Development (Game)",
    "Partnerships (Game)", "Licensing (Game)", "Brand Management (Game)", "IP Management (Game)",
    "Esports Event Management", "Esports Team Management", "Esports Coaching", "Esports Broadcasting",
    "Esports Sponsorship", "Esports Marketing", "Esports Analytics", "Esports Operations",
    "Esports Content Creation", "Esports Journalism", "Esports Law", "Esports Finance", "Esports HR",
    "Esports Business Development", "Esports Partnerships", "Esports Licensing", "Esports Brand Management",
    "Esports IP Management", "Esports Event Planning", "Esports Production", "Esports Broadcasting",
    "Esports Commentating", "Esports Analysis", "Esports Coaching", "Esports Training", "Esports Recruitment",
    "Esports Scouting", "Esports Player Management", "Esports Team Management", "Esports Organization Management",
    "Esports League Management", "Esports Tournament Management", "Esports Venue Management Software",
    "Esports Sponsorship Management Software", "Esports Marketing Automation Software",
    "Esports Content Management Systems", "Esports Social Media Management Tools",
    "Esports PR Tools", "Esports Brand Monitoring Tools", "Esports Community Management Software",
    "Esports Fan Engagement Platforms", "Esports Merchandise Management Software",
    "Esports Ticketing Platforms", "Esports Hospitality Management Software",
    "Esports Logistics Management Software", "Esports Security Management Software",
    "Esports Legal Management Software", "Esports Finance Management Software",
    "Esports HR Management Software", "Esports Business Operations Software",
    "Esports Data Analytics Software", "Esports Performance Analysis Software",
    "Esports Coaching Software", "Esports Training Platforms", "Esports Scouting Tools",
    "Esports Player Databases", "Esports Team Databases", "Esports Organization Databases",
    "Esports League Databases"
])

# --- Functions for Resume Processing ---
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        return text
    except Exception as e:
        log_system_event("ERROR", "PDF_EXTRACTION_FAILED", {"filename": pdf_file.name, "error": str(e)})
        st.error(f"Error extracting text from {pdf_file.name}: {e}")
        return None

def extract_contact_info(text):
    """Extracts email and phone number from text."""
    email_pattern = r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}"
    phone_pattern = r"(?:\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}" # More flexible phone pattern
    
    email = re.search(email_pattern, text)
    phone = re.search(phone_pattern, text)
    
    return email.group(0) if email else "N/A", phone.group(0) if phone else "N/A"

def preprocess_text(text):
    """Cleans and tokenizes text, removing stop words."""
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text) # Remove non-alphabetic characters
    tokens = text.split()
    tokens = [word for word in tokens if word not in STOP_WORDS]
    return " ".join(tokens)

def get_word_cloud_text(text):
    """Generates text for word cloud by filtering for master skills."""
    text = text.lower()
    words = re.findall(r'\b\w+\b', text) # Extract words
    filtered_words = [word for word in words if word in MASTER_SKILLS or any(s.lower() == word for s in MASTER_SKILLS)]
    return " ".join(filtered_words)

def generate_word_cloud(text):
    """Generates and displays a word cloud."""
    if not text.strip():
        st.info("No relevant keywords found for word cloud.")
        return

    wordcloud = WordCloud(width=800, height=400, background_color='white',
                          max_words=100, contour_width=3, contour_color='steelblue').generate(text)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    st.pyplot(fig)
    plt.close(fig) # Close the plot to free memory

def calculate_cosine_similarity(text1, text2):
    """Calculates cosine similarity between two texts using sentence embeddings."""
    if model is None:
        st.warning("Embedding model not loaded. Cannot calculate similarity.")
        return 0.0
    
    embeddings = model.encode([text1, text2])
    similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return similarity

def predict_screening_status(resume_text):
    """Predicts screening status using the loaded ML model."""
    if ml_model is None or model is None:
        return "Model Not Loaded"
    
    try:
        resume_embedding = model.encode([resume_text])
        prediction = ml_model.predict(resume_embedding)
        
        # Assuming the model outputs 0 for Reject, 1 for Interview, 2 for Shortlisted
        # Adjust these mappings based on your actual model's output
        status_map = {0: "Rejected", 1: "Interview", 2: "Shortlisted"}
        predicted_status = status_map.get(prediction[0], "Unknown")
        return predicted_status
    except Exception as e:
        log_system_event("ERROR", "ML_PREDICTION_FAILED", {"error": str(e), "resume_snippet": resume_text[:200]})
        return f"Prediction Error: {e}"

def identify_matched_and_missing_skills(resume_text, job_description_text):
    """Identifies skills from MASTER_SKILLS present in resume and missing from JD."""
    resume_words = set(preprocess_text(resume_text).lower().split())
    jd_words = set(preprocess_text(job_description_text).lower().split())

    matched_skills = [skill for skill in MASTER_SKILLS if skill.lower() in resume_words and skill.lower() in jd_words]
    missing_skills = [skill for skill in MASTER_SKILLS if skill.lower() not in resume_words and skill.lower() in jd_words]
    
    # Also find matched keywords in general (not just MASTER_SKILLS)
    all_matched_keywords = list(resume_words.intersection(jd_words).difference(STOP_WORDS))

    return matched_skills, missing_skills, all_matched_keywords

# Generative AI (Gemini Pro) Function - COMMENTED OUT AS PER USER REQUEST
# def generate_interview_questions(resume_text, job_description, count=5):
#     """Generates interview questions based on resume and job description using Gemini."""
#     if not genai.configure(api_key=st.secrets["GOOGLE_API_KEY"]): # Re-check API key access
#         st.error("Google API Key not configured. Cannot generate interview questions.")
#         return []

#     try:
#         model_genai = genai.GenerativeModel('gemini-pro')
#         prompt = f"""
#         Based on the following resume and job description, generate {count} technical interview questions.
#         Focus on key skills and experiences mentioned in the resume that are relevant to the job description.
#         Prioritize questions that would assess deep understanding and problem-solving abilities.
#         Resume:\n{resume_text}\n\n
#         Job Description:\n{job_description}\n\n
#         Format the output as a numbered list of questions.
#         """
#         response = model_genai.generate_content(prompt)
#         questions = response.text.strip().split('\n')
#         return [q for q in questions if q.strip()] # Filter out empty lines
#     except Exception as e:
#         st.error(f"Error generating interview questions: {e}")
#         log_system_event("ERROR", "GEMINI_QUESTION_GENERATION_FAILED", {"error": str(e)})
#         return ["Could not generate questions. Please check API key and try again."]

def process_resumes(uploaded_files, job_description, required_score_threshold=0.5):
    """Processes uploaded resumes against a job description."""
    if model is None or ml_model is None:
        st.error("Model not loaded. Please check the model files and try again.")
        return pd.DataFrame()

    job_description_processed = preprocess_text(job_description)
    job_description_embedding = model.encode([job_description_processed])

    results = []
    total_files = len(uploaded_files)
    processed_count = 0

    log_user_action(st.session_state.user_email, "RESUME_SCREENING_STARTED", {"job_description_length": len(job_description)})

    progress_bar = st.progress(0)
    status_text = st.empty()

    for uploaded_file in uploaded_files:
        status_text.text(f"Processing {uploaded_file.name}...")
        resume_text = extract_text_from_pdf(uploaded_file)
        if resume_text:
            name_match = re.search(r"^[A-Z][a-zA-Z.\s]+", resume_text.split('\n')[0])
            candidate_name = name_match.group(0).strip() if name_match else uploaded_file.name.replace(".pdf", "")
            
            email, phone = extract_contact_info(resume_text)
            
            resume_processed = preprocess_text(resume_text)
            resume_embedding = model.encode([resume_processed])
            
            similarity = calculate_cosine_similarity(resume_processed, job_description_processed)
            
            # Predict screening status using the ML model
            predicted_status = predict_screening_status(resume_embedding.tolist()[0]) # Pass the raw embedding list

            # Generate word cloud data
            word_cloud_text_for_resume = get_word_cloud_text(resume_text)

            # Identify matched and missing skills
            matched_skills, missing_skills, all_matched_keywords = identify_matched_and_missing_skills(resume_text, job_description)

            # Determine match level
            match_level = "High" if similarity >= required_score_threshold else "Low"

            results.append({
                "Resume Name": uploaded_file.name,
                "Candidate Name": candidate_name,
                "Email": email,
                "Phone": phone,
                "Similarity Score": f"{similarity:.2f}",
                "Predicted Status": predicted_status,
                "Match Level": match_level,
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                "Keywords in JD & Resume": ", ".join(all_matched_keywords) if all_matched_keywords else "None",
                "Resume Text": resume_text, # Keep original text for display/download
                "WordCloudText": word_cloud_text_for_for_resume # Text for generating word cloud later
            })
            log_user_action(st.session_state.user_email, "RESUME_SCREENED", {
                "resume_name": uploaded_file.name,
                "similarity_score": float(f"{similarity:.2f}"),
                "predicted_status": predicted_status,
                "match_level": match_level,
                "jd_id": st.session_state.get('selected_jd_id', 'N/A') # Assuming JD ID is stored
            })
            update_metrics_summary(st.session_state.user_email, "total_resumes_screened", 1)
            update_metrics_summary(st.session_state.user_email, "user_resumes_screened", 1)
        else:
            log_system_event("WARNING", "RESUME_SKIPPED", {"filename": uploaded_file.name, "reason": "Text extraction failed"})
            st.warning(f"Skipping {uploaded_file.name} due to text extraction error.")

        processed_count += 1
        progress_bar.progress(processed_count / total_files)

    status_text.text("Resume screening complete!")
    progress_bar.empty() # Clear the progress bar after completion
    return pd.DataFrame(results)

def screener_page():
    st.markdown('<div class="dashboard-header">🚀 Resume Screener</div>', unsafe_allow_html=True)

    if model is None or ml_model is None:
        st.warning("Application models are not loaded. Please ensure `ml_screening_model.pkl` is available.")
        return

    # Initialize session state for storing job description and results
    if "job_description" not in st.session_state:
        st.session_state.job_description = ""
    if "screening_results" not in st.session_state:
        st.session_state.screening_results = pd.DataFrame()
    if "display_detail_for_resume" not in st.session_state:
        st.session_state.display_detail_for_resume = None

    # Job Description Input
    st.subheader("1. Enter Job Description")
    job_description = st.text_area(
        "Paste the Job Description here:",
        value=st.session_state.job_description,
        height=200,
        key="job_description_input"
    )
    # Update session state whenever the text area changes
    st.session_state.job_description = job_description

    st.subheader("2. Upload Resumes (PDFs only)")
    uploaded_files = st.file_uploader(
        "Choose PDF resume files", 
        type="pdf", 
        accept_multiple_files=True, 
        key="resume_uploader"
    )
    
    if uploaded_files:
        log_user_action(st.session_state.user_email, "RESUMES_UPLOADED", {"count": len(uploaded_files)})
        st.info(f"{len(uploaded_files)} files uploaded. Click 'Screen Resumes' to start.")

    # Screening Settings
    st.subheader("3. Screening Settings")
    required_score_threshold = st.slider(
        "Minimum Similarity Score for 'High Match':",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.01,
        help="Resumes with a similarity score equal to or above this threshold will be flagged as 'High Match'."
    )

    # Screen Button
    if st.button("🚀 Screen Resumes", type="primary", use_container_width=True, key="screen_button"):
        if not job_description:
            st.error("Please enter a job description.")
            log_user_action(st.session_state.user_email, "SCREENING_INITIATED_FAILED", {"reason": "No job description"})
        elif not uploaded_files:
            st.error("Please upload at least one resume.")
            log_user_action(st.session_state.user_email, "SCREENING_INITIATED_FAILED", {"reason": "No resumes uploaded"})
        else:
            log_user_action(st.session_state.user_email, "SCREENING_INITIATED_SUCCESS", {"num_resumes": len(uploaded_files), "threshold": required_score_threshold})
            with st.spinner("Screening resumes... This may take a moment."):
                df_results = process_resumes(uploaded_files, job_description, required_score_threshold)
                st.session_state.screening_results = df_results
                st.success("Screening complete! See results below.")
                st.session_state.display_detail_for_resume = None # Reset detail view

    # Display Results
    if not st.session_state.screening_results.empty:
        st.markdown("---")
        st.subheader("4. Screening Results")
        
        df_display = st.session_state.screening_results.copy()
        # Create a "View Details" button column
        df_display['Action'] = [f"View Details_{i}" for i in range(len(df_display))]
        
        # Make the dataframe interactive for button clicks
        # We need to handle button clicks outside the dataframe loop to avoid reruns
        
        # Sort results for better visualization
        sort_column = st.selectbox("Sort results by:", options=['Similarity Score', 'Predicted Status', 'Candidate Name'], index=0)
        sort_order = st.radio("Sort order:", options=['Descending', 'Ascending'], index=0, horizontal=True)

        if sort_column == 'Similarity Score':
            df_display['Similarity Score'] = pd.to_numeric(df_display['Similarity Score'])
            st.session_state.screening_results = st.session_state.screening_results.sort_values(
                by='Similarity Score', 
                ascending=(sort_order == 'Ascending')
            ).reset_index(drop=True)
            df_display = df_display.sort_values(by='Similarity Score', ascending=(sort_order == 'Ascending')).reset_index(drop=True)
        else:
            st.session_state.screening_results = st.session_state.screening_results.sort_values(
                by=sort_column, 
                ascending=(sort_order == 'Ascending')
            ).reset_index(drop=True)
            df_display = df_display.sort_values(by=sort_column, ascending=(sort_order == 'Ascending')).reset_index(drop=True)


        st.dataframe(
            df_display[['Candidate Name', 'Similarity Score', 'Predicted Status', 'Match Level', 'Action']],
            hide_index=True,
            use_container_width=True,
            column_config={
                "Action": st.column_config.Button(
                    label="View Details",
                    help="Click to see full details and actions for this resume.",
                    key="view_details_button_col"
                )
            }
        )

        # Handle 'View Details' button clicks
        for i, row in df_display.iterrows():
            if st.session_state.get(f"view_details_button_col:{i}"):
                st.session_state.display_detail_for_resume = i
                break # Exit loop once a button is clicked to prevent multiple triggers

        # Display detailed view if a resume is selected
        if st.session_state.display_detail_for_resume is not None:
            idx = st.session_state.display_detail_for_resume
            candidate_row = st.session_state.screening_results.iloc[idx]
            
            st.markdown("---")
            st.subheader(f"Detailed View for {candidate_row['Candidate Name']}")

            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.write(f"**Similarity Score:** {candidate_row['Similarity Score']}")
                st.write(f"**Predicted Status:** {candidate_row['Predicted Status']}")
                st.write(f"**Match Level:** {candidate_row['Match Level']}")
            with col_info2:
                st.write(f"**Email:** {candidate_row['Email']}")
                st.write(f"**Phone:** {candidate_row['Phone']}")
                # Create a mailto link
                email_subject = urllib.parse.quote(f"Regarding your application for [Job Title] - {candidate_row['Candidate Name']}")
                email_body = urllib.parse.quote("Dear [Candidate Name],\n\n...")
                mailto_link = f"mailto:{candidate_row['Email']}?subject={email_subject}&body={email_body}"
                st.markdown(f"**[Contact Candidate via Email]({mailto_link})**")


            st.markdown("---")
            st.write("#### Skills & Keywords Analysis")
            st.write(f"**Matched Skills (from Master List):** {candidate_row['Matched Skills']}")
            st.write(f"**Missing Skills (from Master List):** {candidate_row['Missing Skills']}")
            st.write(f"**Keywords found in both JD & Resume:** {candidate_row['Keywords in JD & Resume']}")

            st.write("#### Resume Word Cloud (Skills/Keywords)")
            generate_word_cloud(candidate_row["WordCloudText"])

            st.write("#### Full Resume Text Preview")
            with st.expander("Click to view full resume text"):
                st.text_area(f"Resume Text for {candidate_row['Candidate Name']}", 
                             candidate_row['Resume Text'], height=300, disabled=True)

            # Generative AI (Gemini Pro) integration - COMMENTED OUT AS PER USER REQUEST
            # st.markdown("---")
            # st.write("#### AI-Powered Interview Questions (Generated by Gemini Pro)")
            # if st.button(f"Generate Questions for {candidate_row['Candidate Name']}", key=f"gen_q_{idx}"):
            #     with st.spinner("Generating interview questions..."):
            #         questions = generate_interview_questions(candidate_row['Resume Text'], job_description)
            #         for i, q in enumerate(questions):
            #             st.write(f"{i+1}. {q}")
            #     log_user_action(st.session_state.user_email, "INTERVIEW_QUESTIONS_GENERATED", {"candidate_name": candidate_row['Candidate Name'], "num_questions": len(questions)})


            # Option to download individual resume
            st.markdown("---")
            st.write("#### Download Resume")
            pdf_name = candidate_row['Resume Name']
            # Find the original uploaded file object to get its bytes
            original_file_obj = next((f for f in uploaded_files if f.name == pdf_name), None)

            if original_file_obj:
                original_file_obj.seek(0) # Go to the beginning of the file-like object
                st.download_button(
                    label=f"Download {pdf_name}",
                    data=original_file_obj.read(),
                    file_name=pdf_name,
                    mime="application/pdf",
                    key=f"download_{idx}"
                )
            else:
                st.warning(f"Original file '{pdf_name}' not found for direct download.")
            
            st.button("↩️ Back to Results Table", key="back_to_results", 
                      on_click=lambda: st.session_state.update(display_detail_for_resume=None))

        # Download All Results
        st.markdown("---")
        st.subheader("5. Download All Results")
        csv_data = st.session_state.screening_results.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download All Screening Results as CSV",
            data=csv_data,
            file_name="resume_screening_results.csv",
            mime="text/csv",
            key="download_all_csv"
        )
    else:
        st.info("Upload resumes and enter a job description to see screening results here.")

# Entry point for the screener module when imported by main.py
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Resume Screener Pro")
    st.title("Resume Screener Pro (Standalone Test)")
    # Mock user session state for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_user@example.com")
    
    resume_screener_page()
