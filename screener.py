print("DEBUG: screener.py is being loaded.") # Diagnostic print to confirm file is being run/imported

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
import traceback # Added for detailed error logging
from io import BytesIO # Added: Required for reading PDF in-memory
from sklearn.feature_extraction.text import TfidfVectorizer # Added: Required for TF-IDF calculation


# Import logging functions
from utils.logger import log_user_action, update_metrics_summary, log_system_event
# Assuming utils.config exists, if not, remove this line or create the file
# from utils.config import load_config

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
    log_system_event("INFO", "NLTK_DOWNLOAD", {"resource": "stopwords"}) # Log NLTK download

# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        # Ensure ml_screening_model.pkl exists before loading
        if not os.path.exists("ml_screening_model.pkl"):
            # Provide a more user-friendly message if the model file is missing
            st.error("Model file 'ml_screening_model.pkl' not found. Please ensure it's in the same directory as this script.")
            raise FileNotFoundError("ml_screening_model.pkl not found.")
        ml_model = joblib.load("ml_screening_model.pkl")
        log_system_event("INFO", "ML_MODEL_LOADED", {"model_name": "all-MiniLM-L6-v2", "ml_model_file": "ml_screening_model.pkl"})
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
        log_system_event("ERROR", "ML_MODEL_LOAD_FAILED", {"error": str(e), "traceback": traceback.format_exc()})
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
# Keep this set empty if you want the system to use its default stop word filtering.
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
STOP_WORDS = NLTK_STOP_WORDS.union(CUSTOM_STOP_WORDS)

# --- Page Styling ---
st.markdown("""
<style>
.st-emotion-cache-1czw38s {
    padding-top: 10px;
}
.screener-container {
    padding: 2rem;
    background: rgba(255, 255, 255, 0.95);
    border-radius: 20px;
    box-shadow: 0 10px 30px rgba(0,0,0,0.08);
    animation: fadeInSlide 0.7s ease-in-out;
    margin-bottom: 2rem;
}
@keyframes fadeInSlide {
    0% { opacity: 0; transform: translateY(20px); }
    100% { opacity: 1; transform: translateY(0); }
}
h2 {
    color: #00cec9;
    font-weight: 700;
}
.stMetric {
    background-color: #f0f2f6;
    border-radius: 10px;
    padding: 1rem;
    margin-bottom: 1rem;
}
.stProgress > div > div > div > div {
    background-color: #00cec9 !important;
}
</style>
""", unsafe_allow_html=True)

# --- Helper Functions ---
@st.cache_data(show_spinner=False)
def extract_text_from_pdf(pdf_file):
    """Extracts text from a PDF file using pdfplumber."""
    try:
        with pdfplumber.open(pdf_file) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)
        return text
    except Exception as e:
        log_system_event("ERROR", "PDF_EXTRACTION_FAILED", {"file": pdf_file.name, "error": str(e), "traceback": traceback.format_exc()})
        return None

def extract_years_of_experience(text):
    """
    Extracts years of experience from text.
    Looks for patterns like 'X years experience', 'X+ years', 'experience of X years'.
    Returns the maximum number of years found.
    """
    text = text.lower()
    total_months = 0
    job_date_ranges = re.findall(
        r'(\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})\s*(?:to|–|-)\s*(present|\b(?:jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+\d{4})',
        text
    )

    for start, end in job_date_ranges:
        try:
            start_date = datetime.strptime(start.strip(), '%b %Y')
        except ValueError:
            try:
                start_date = datetime.strptime(start.strip(), '%B %Y')
                # If month is full name, try with full name
            except ValueError:
                continue

        if end.strip() == 'present':
            end_date = datetime.now()
        else:
            try:
                end_date = datetime.strptime(end.strip(), '%b %Y')
            except ValueError:
                try:
                    end_date = datetime.strptime(end.strip(), '%B %Y')
                except ValueError:
                    continue

        delta_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        total_months += max(delta_months, 0)

    if total_months == 0:
        # Fallback to simple regex if date range extraction fails
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                log_system_event("WARNING", "EXPERIENCE_PARSE_ERROR", {"text_snippet": text[:50], "error": "Could not convert to float"})
                return 0.0
    return round(total_months / 12, 1)

def extract_contact_info(text):
    """Extracts email and phone number using regex."""
    email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text)
    phone_match = re.search(r'(\+?\d{1,3}[-.\s]?)?\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}', text)
    return email_match.group(0) if email_match else "N/A", phone_match.group(0) if phone_match else "N/A"

def generate_ai_suggestion(score, years_exp, missing_skills, required_skills):
    """Generates an AI-like suggestion based on screening criteria."""
    # Retrieve cutoff values from session state, with defaults
    cutoff_score = st.session_state.get('screening_cutoff_score', 75)
    min_exp_required = st.session_state.get('screening_min_experience', 2)

    if score >= 90 and years_exp >= min_exp_required:
        return "Strongly Recommended for Interview: Excellent match!"
    elif score >= cutoff_score and years_exp >= min_exp_required:
        return "Recommended for Interview: Good potential."
    elif score >= 60 and years_exp < min_exp_required and not missing_skills:
        return "Consider for Junior Role/Training: High score, but less experience."
    elif score >= 60 and len(missing_skills) <= len(required_skills) / 3:
        return "Potential Match with Skill Gap: Requires review of missing skills."
    else:
        return "Not a Direct Match: Consider for other roles or re-skill."

def clean_text_for_wordcloud(text):
    """Basic cleaning for word cloud to remove non-alphanumeric and extra spaces."""
    # Remove special characters, numbers, and convert to lowercase
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Remove common resume sections or generic words that won't add value
    stop_words = ["experience", "years", "skills", "education", "project", "work", "roles", "description", "responsibilities", "knowledge", "ability", "developed", "used", "proficient", "strong"]
    words = text.lower().split()
    cleaned_words = [word for word in words if word not in stop_words and len(word) > 2]
    return " ".join(cleaned_words)

def clean_text(text):
    """Cleans text by removing newlines, extra spaces, and non-ASCII characters."""
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\x00-\x7F]+', ' ', text)
    return text.strip().lower()

def extract_relevant_keywords(text, filter_set):
    """
    Extracts relevant keywords from text, prioritizing multi-word skills from filter_set.
    If filter_set is empty, it falls back to filtering out general STOP_WORDS.
    """
    cleaned_text = clean_text(text)
    extracted_keywords = set()

    if filter_set: # If a specific filter_set (like MASTER_SKILLS) is provided
        # Sort skills by length descending to match longer phrases first
        sorted_filter_skills = sorted(list(filter_set), key=len, reverse=True)
        
        temp_text = cleaned_text # Use a temporary text to remove matched phrases

        for skill_phrase in sorted_filter_skills:
            # Create a regex pattern to match the whole skill phrase
            # \b ensures whole word match, re.escape handles special characters in skill names
            pattern = r'\b' + re.escape(skill_phrase.lower()) + r'\b'
            
            # Find all occurrences of the skill phrase
            matches = re.findall(pattern, temp_text)
            if matches:
                extracted_keywords.add(skill_phrase.lower()) # Add the original skill (lowercase)
                # Replace the found skill with placeholders to avoid re-matching parts of it
                temp_text = re.sub(pattern, " ", temp_text)
        
        # After extracting phrases, now extract individual words that are in the filter_set
        # and haven't been part of a multi-word skill already extracted.
        # This ensures single-word skills from MASTER_SKILLS are also captured.
        individual_words_remaining = set(re.findall(r'\b\w+\b', temp_text))
        for word in individual_words_remaining:
            if word in filter_set:
                extracted_keywords.add(word)

    else: # Fallback: if no specific filter_set (MASTER_SKILLS is empty), use the default STOP_WORDS logic
        all_words = set(re.findall(r'\b\w+\b', cleaned_text))
        extracted_keywords = {word for word in all_words if word not in STOP_WORDS}

    return extracted_keywords

def extract_text_from_pdf(uploaded_file):
    """Extracts text from an uploaded PDF file."""
    try:
        with pdfplumber.open(uploaded_file) as pdf:
            text = ''.join(page.extract_text() or '' for page in pdf.pages)
        log_system_event("INFO", "PDF_TEXT_EXTRACTED", {"filename": uploaded_file.name, "text_length": len(text)})
        return text
    except Exception as e:
        log_system_event("ERROR", "PDF_EXTRACTION_FAILED", {"filename": uploaded_file.name, "error": str(e), "traceback": traceback.format_exc()})
        return f"[ERROR] {str(e)}"

def extract_email(text):
    """Extracts an email address from the given text."""
    match = re.search(r'[\w\.-]+@[\w\.-]+\.\w+', text)
    return match.group(0) if match else None

def extract_name(text):
    """
    Attempts to extract a name from the first few lines of the resume text.
    This is a heuristic and might not be perfect for all resume formats.
    """
    lines = text.strip().split('\n')
    if not lines:
        return None

    potential_name_lines = []
    for line in lines[:3]:
        line = line.strip()
        # Refined regex to be more robust for names, avoiding lines with too many non-alpha chars
        if not re.search(r'[@\d\.\-]', line) and len(line.split()) <= 4 and (line.isupper() or (line and line[0].isupper() and all(word[0].isupper() or not word.isalpha() for word in line.split()))):
            potential_name_lines.append(line)

    if potential_name_lines:
        name = max(potential_name_lines, key=len)
        name = re.sub(r'summary|education|experience|skills|projects|certifications', '', name, flags=re.IGNORECASE).strip()
        if name:
            return name.title()
    return None

# --- Concise AI Suggestion Function (for table display) ---
@st.cache_data(show_spinner="Generating concise AI Suggestion...")
def generate_concise_ai_suggestion(candidate_name, score, years_exp, semantic_similarity):
    """
    Generates a concise AI suggestion based on rules, focusing on overall fit and key points.
    """
    overall_fit_description = ""
    review_focus_text = ""

    if score >= 85 and years_exp >= st.session_state.get('screening_min_experience', 2) and semantic_similarity >= 0.75:
        overall_fit_description = "High alignment with job requirements."
        review_focus_text = "Focus on cultural fit and specific project contributions."
    elif score >= st.session_state.get('screening_cutoff_score', 75) and years_exp >= st.session_state.get('screening_min_experience', 2) and semantic_similarity >= 0.4:
        overall_fit_description = "Moderate fit; good potential."
        review_focus_text = "Probe depth of experience and application of skills."
    else:
        overall_fit_description = "Limited alignment with core requirements."
        review_focus_text = "Consider only if pipeline is limited; focus on foundational skills."

    summary_text = f"**Overall Fit:** {overall_fit_description} **Review Focus:** {review_focus_text}"
    return summary_text

# --- Detailed HR Assessment Function (for top candidate display) ---
@st.cache_data(show_spinner="Generating detailed HR Assessment...")
def generate_detailed_hr_assessment(candidate_name, score, years_exp, semantic_similarity, jd_text, resume_text):
    """
    Generates a detailed, multi-paragraph HR assessment for a candidate.
    """
    assessment_parts = []
    overall_assessment_title = ""
    next_steps_focus = ""

    # Retrieve cutoff values from session state, with defaults
    cutoff_score = st.session_state.get('screening_cutoff_score', 75)
    min_exp_required = st.session_state.get('screening_min_experience', 2)

    # Tier 1: Exceptional Candidate
    if score >= 90 and years_exp >= min_exp_required + 3 and semantic_similarity >= 0.85: # Higher bar for exceptional
        overall_assessment_title = "Exceptional Candidate: Highly Aligned with Strategic Needs"
        assessment_parts.append(f"**{candidate_name}** presents an **exceptional profile** with a high score of {score:.2f}% and {years_exp:.1f} years of experience. This demonstrates a profound alignment with the job description's core requirements, further evidenced by a strong semantic similarity of {semantic_similarity:.2f}.")
        assessment_parts.append("This candidate possesses a robust skill set directly matching critical keywords in the JD, suggesting immediate productivity and minimal ramp-up time. Their extensive experience indicates a capacity for leadership and handling complex challenges. They are poised to make significant contributions from day one.")
        next_steps_focus = "The next steps should focus on assessing cultural integration, exploring leadership potential, and delving into strategic contributions during the interview. This candidate appears to be a strong fit for a pivotal role within the organization."
    # Tier 2: Strong Candidate
    elif score >= 80 and years_exp >= min_exp_required and semantic_similarity >= 0.7:
        overall_assessment_title = "Strong Candidate: Excellent Potential for Key Contributions"
        assessment_parts.append(f"**{candidate_name}** is a **strong candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. They show excellent alignment with the job description, supported by a solid semantic similarity of {semantic_similarity:.2f}.")
        assessment_parts.append("Key strengths include a significant overlap in required skills and practical experience that directly addresses the job's demands. This individual is likely to integrate well and contribute effectively from an early stage, bringing valuable expertise to the team.")
        next_steps_focus = "During the interview, explore specific project methodologies, problem-solving approaches, and long-term career aspirations to confirm alignment with team dynamics and growth opportunities within the company."
    # Tier 3: Promising Candidate
    elif score >= cutoff_score and years_exp >= min_exp_required - 1 and semantic_similarity >= 0.35: # Slightly lower exp for promising
        overall_assessment_title = "Promising Candidate: Requires Focused Review on Specific Gaps"
        assessment_parts.append(f"**{candidate_name}** is a **promising candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. While demonstrating a foundational understanding (semantic similarity: {semantic_similarity:.2f}), there are areas that warrant deeper investigation to ensure a complete fit.")
        
        gaps = []
        if score < cutoff_score + 5: # If score is just above cutoff
            gaps.append("The overall score suggests some core skill areas may need development or further clarification.")
        if years_exp < min_exp_required:
            gaps.append(f"Experience ({years_exp:.1f} yrs) is on the lower side for the role; assess their ability to scale up quickly and take on more responsibility.")
        if semantic_similarity < 0.5:
            gaps.append("Semantic understanding of the JD's nuances might be limited; probe their theoretical knowledge versus practical application in real-world scenarios.")
        
        if gaps:
            assessment_parts.append("Areas for further exploration include: " + " ".join(gaps))
        
        next_steps_focus = "The interview should focus on validating foundational skills, understanding their learning agility, and assessing their potential for growth within the role. Be prepared to discuss specific examples of how they've applied relevant skills and how they handle challenges."
    # Tier 4: Limited Match
    else:
        overall_assessment_title = "Limited Match: Consider Only for Niche Needs or Pipeline Building"
        assessment_parts.append(f"**{candidate_name}** shows a **limited match** with a score of {score:.2f}% and {years_exp:.1f} years of experience (semantic similarity: {semantic_similarity:.2f}). This profile indicates a significant deviation from the core requirements of the job description.")
        assessment_parts.append("Key concerns include a low overlap in essential skills and potentially insufficient experience for the role's demands. While some transferable skills may exist, a substantial investment in training or a re-evaluation of role fit would likely be required for this candidate to succeed.")
        next_steps_focus = "This candidate is generally not recommended for the current role unless there are specific, unforeseen niche requirements or a strategic need to broaden the candidate pool significantly. If proceeding, focus on understanding their fundamental capabilities and long-term career aspirations."

    final_assessment = f"**Overall HR Assessment: {overall_assessment_title}**\n\n"
    final_assessment += "\n".join(assessment_parts) + "\n\n"
    final_assessment += f"**Recommended Interview Focus & Next Steps:** {next_steps_focus}"

    return final_assessment


def semantic_score(resume_text, jd_text, years_exp):
    """
    Calculates a semantic score using an ML model and provides additional details.
    Falls back to smart_score if the ML model is not loaded or prediction fails.
    Applies STOP_WORDS filtering for keyword analysis (internally, not for display).
    """
    jd_clean = clean_text(jd_text)
    resume_clean = clean_text(resume_text)

    score = 0.0
    feedback = "Initial assessment." # This will be overwritten by the generate_concise_ai_suggestion function
    semantic_similarity = 0.0

    if ml_model is None or model is None:
        log_system_event("WARNING", "ML_MODELS_NOT_LOADED_FOR_SEMANTIC_SCORE", {"reason": "Falling back to basic score"})
        # Removed st.warning here as it's handled by load_ml_model
        # Simplified fallback for score and feedback
        resume_words = extract_relevant_keywords(resume_clean, MASTER_SKILLS if MASTER_SKILLS else STOP_WORDS)
        jd_words = extract_relevant_keywords(jd_clean, MASTER_SKILLS if MASTER_SKILLS else STOP_WORDS)
        
        overlap_count = len(resume_words.intersection(jd_words))
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)
        
        feedback = "Due to missing ML models, a detailed AI suggestion cannot be provided. Basic score derived from keyword overlap. Manual review is highly recommended."
        
        return score, feedback, 0.0 # Return 0 for semantic similarity if ML not available


    try:
        jd_embed = model.encode(jd_clean)
        resume_embed = model.encode(resume_clean)

        semantic_similarity = cosine_similarity(jd_embed.reshape(1, -1), resume_embed.reshape(1, -1))[0][0]
        semantic_similarity = float(np.clip(semantic_similarity, 0, 1))

        # Internal calculation for model, not for display
        # Use the new extraction logic for model features
        resume_words_filtered = extract_relevant_keywords(resume_clean, MASTER_SKILLS if MASTER_SKILLS else STOP_WORDS)
        jd_words_filtered = extract_relevant_keywords(jd_clean, MASTER_SKILLS if MASTER_SKILLS else STOP_WORDS)
        keyword_overlap_count = len(resume_words_filtered.intersection(jd_words_filtered))
        
        years_exp_for_model = float(years_exp) if years_exp is not None else 0.0

        features = np.concatenate([jd_embed, resume_embed, [years_exp_for_model], [keyword_overlap_count]])

        predicted_score = ml_model.predict([features])[0]

        if len(jd_words_filtered) > 0:
            jd_coverage_percentage = (keyword_overlap_count / len(jd_words_filtered)) * 100
        else:
            jd_coverage_percentage = 0.0

        blended_score = (predicted_score * 0.6) + \
                        (jd_coverage_percentage * 0.1) + \
                        (semantic_similarity * 100 * 0.3)

        if semantic_similarity > 0.7 and years_exp >= 3:
            blended_score += 5

        score = float(np.clip(blended_score, 0, 100))
        
        # The AI suggestion text will be generated separately for display by generate_concise_ai_suggestion.
        return round(score, 2), "AI suggestion will be generated...", round(semantic_similarity, 2) # Placeholder feedback


    except Exception as e:
        log_system_event("ERROR", "SEMANTIC_SCORE_CALC_FAILED", {"error": str(e), "traceback": traceback.format_exc()})
        st.warning(f"Error during semantic scoring, falling back to basic: {e}")
        # Simplified fallback for score and feedback if ML prediction fails
        # Use the new extraction logic for fallback
        resume_words = extract_relevant_keywords(resume_clean, MASTER_SKILLS if MASTER_SKILLS else STOP_WORDS)
        jd_words = extract_relevant_keywords(jd_clean, MASTER_SKILLS if MASTER_SKILLS else STOP_WORDS)
        
        overlap_count = len(resume_words.intersection(jd_words))
        total_jd_words = len(jd_words)
        
        basic_score = (overlap_count / total_jd_words) * 70 if total_jd_words > 0 else 0
        basic_score += min(years_exp * 5, 30) # Add up to 30 for experience
        score = round(min(basic_score, 100), 2)

        feedback = "Due to an error in core AI model, a detailed AI suggestion cannot be provided. Basic score derived. Manual review is highly recommended."

        return score, feedback, 0.0 # Return 0 for semantic similarity on fallback


# --- Email Generation Function ---
def create_mailto_link(recipient_email, candidate_name, job_title="Job Opportunity", sender_name="Recruiting Team"):
    """
    Generates a mailto: link with pre-filled subject and body for inviting a candidate.
    """
    subject = urllib.parse.quote(f"Invitation for Interview - {job_title} - {candidate_name}")
    body = urllib.parse.quote(f"""Dear {candidate_name},

We were very impressed with your profile and would like to invite you for an interview for the {job_title} position.

Best regards,

The {sender_name}""")
    return f"mailto:{recipient_email}?subject={subject}&body={body}"

# --- Function to encapsulate the Resume Screener logic ---
def resume_screener_page():
    print("DEBUG: resume_screener_page function is being called/defined.") # Diagnostic print
    # Ensure user is logged in
    if 'user_email' not in st.session_state:
        st.warning("Please log in to use the Resume Screener.")
        log_user_action("unauthenticated", "RESUME_SCREENER_ACCESS_DENIED", {"reason": "Not logged in"})
        return

    user_email = st.session_state.user_email
    log_user_action(user_email, "RESUME_SCREENER_PAGE_ACCESSED")

    st.markdown('<div class="screener-container">', unsafe_allow_html=True)
    st.markdown("## 🧠 AI Resume Screener")
    st.caption("Upload resumes, provide a Job Description, and let the AI analyze the match.")

    # --- Job Description Input ---
    st.markdown("### 📝 Job Description")
    # Ensure 'data' directory exists for JD files
    jd_folder = "data"
    os.makedirs(jd_folder, exist_ok=True)
    jd_files = [f for f in os.listdir(jd_folder) if f.endswith(".txt")]
    
    jd_options = ["Paste Manually"] + sorted(jd_files)
    jd_source = st.radio("Choose JD Source:", jd_options, key="jd_source_radio")

    job_description_text = ""
    if jd_source == "Paste Manually":
        job_description_text = st.text_area("Paste Job Description here:", height=200, key="manual_jd")
        if job_description_text:
            log_user_action(user_email, "JD_ENTERED_MANUALLY", {"jd_length": len(job_description_text)})
    else:
        jd_file_path = os.path.join(jd_folder, jd_source)
        try:
            with open(jd_file_path, "r", encoding="utf-8") as f:
                job_description_text = f.read()
            st.text_area(f"Content of {jd_source}:", value=job_description_text, height=200, disabled=True, key="loaded_jd")
            log_user_action(user_email, "JD_LOADED_FROM_FILE", {"file_name": jd_source, "jd_length": len(job_description_text)})
        except Exception as e:
            st.error(f"Error loading JD from file: {e}. Please ensure the file is valid and accessible.")
            log_system_event("ERROR", "JD_FILE_LOAD_FAILED_SCREENER", {"user_email": user_email, "file": jd_source, "error": str(e), "traceback": traceback.format_exc()})
            job_description_text = ""

    if not job_description_text:
        st.warning("Please provide or select a Job Description to proceed.")
        st.stop()

    # --- Skills and Experience Input ---
    st.markdown("### 🎯 Screening Criteria")
    col1, col2 = st.columns(2)
    with col1:
        required_skills_input = st.text_area(
            "Required Skills (comma-separated, e.g., Python, SQL, AWS)",
            "Python, SQL, Data Analysis, Machine Learning",
            key="required_skills"
        )
    with col2:
        min_experience = st.number_input(
            "Minimum Years of Experience Required",
            min_value=0, value=2, step=1, key="min_experience"
        )
        # Store for analytics/email page
        st.session_state['screening_min_experience'] = min_experience
        
        # Add a cutoff score input for screening
        cutoff_score = st.slider(
            "Minimum Similarity Score (%) for Shortlisting",
            min_value=0, max_value=100, value=75, step=1, key="cutoff_score_slider"
        )
        st.session_state['screening_cutoff_score'] = cutoff_score # Store for email page


    required_skills = [skill.strip().lower() for skill in required_skills_input.split(',') if skill.strip()]
    if not required_skills:
        st.warning("Please enter at least one required skill.")
        log_user_action(user_email, "SCREENING_CRITERIA_MISSING_SKILLS", {"reason": "No required skills entered"})
        st.stop()

    # --- Resume Upload ---
    st.markdown("### 📄 Upload Resumes")
    uploaded_resumes = st.file_uploader(
        "Upload Resume PDFs (multiple allowed)",
        type="pdf",
        accept_multiple_files=True,
        key="resume_uploads"
    )

    if uploaded_resumes and st.button("🚀 Start Screening"):
        st.session_state['screening_results'] = pd.DataFrame() # Clear previous results
        results = []
        jd_text_lower = job_description_text.lower()
        # jd_words_set = set(re.findall(r'\b\w+\b', jd_text_lower)) # Words from JD - not directly used in core logic but good for analysis

        progress_text = "Operation in progress. Please wait."
        my_bar = st.progress(0, text=progress_text)
        # Initialize status_text here
        status_text = st.empty()
        
        log_user_action(user_email, "SCREENING_STARTED", {
            "num_resumes": len(uploaded_resumes),
            "jd_source": jd_source,
            "min_experience_req": min_experience,
            "required_skills_count": len(required_skills),
            "shortlighting_cutoff_score": cutoff_score
        })
        update_metrics_summary("total_screenings_run", 1)
        update_metrics_summary("user_screenings_run", 1, user_email=user_email)

        for i, resume_file in enumerate(uploaded_resumes):
            status_text.text(f"Processing {resume_file.name} ({i+1}/{len(uploaded_resumes)})...")
            my_bar.progress((i + 1) / len(uploaded_resumes))

            # Read PDF in-memory as BytesIO
            pdf_bytes = BytesIO(resume_file.read())
            resume_text = extract_text_from_pdf(pdf_bytes)

            if resume_text.startswith("[ERROR]"): # Check for error string from extract_text_from_pdf
                st.error(f"Failed to process {resume_file.name}: {resume_text.replace('[ERROR] ', '')}. Skipping...")
                log_system_event("WARNING", "RESUME_SKIPPED_DUE_TO_PARSE_ERROR", {"user_email": user_email, "resume_name": resume_file.name, "error_detail": resume_text})
                continue

            # Basic Information Extraction
            # Improved candidate name extraction: look for common patterns at the beginning
            candidate_name_match = re.match(r'^(?:Mr\.|Ms\.|Dr\.)?\s*([A-Za-z\s.-]{2,})', resume_text.strip())
            candidate_name = candidate_name_match.group(1).strip() if candidate_name_match else resume_file.name.replace(".pdf", "").replace("_", " ").title()
            
            email = extract_email(resume_text)
            phone = extract_contact_info(resume_text)[1] # Get phone from tuple
            years_experience = extract_years_of_experience(resume_text)
            
            # Skill Matching
            resume_text_lower = resume_text.lower()
            matched_skills = [skill for skill in required_skills if skill in resume_text_lower]
            missing_skills = [skill for skill in required_skills if skill not in resume_text_lower]

            # Similarity Score (Cosine Similarity with TF-IDF)
            documents = [jd_text_lower, resume_text_lower]
            try:
                # Add 'stop_words' to TfidfVectorizer for better relevance
                vectorizer = TfidfVectorizer(stop_words='english')
                tfidf_matrix = vectorizer.fit_transform(documents)
                cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0] 
                similarity_score_percent = round(cosine_sim * 100, 2)
            except Exception as e:
                similarity_score_percent = 0.0 # Default to 0 if vectorization fails
                log_system_event("ERROR", "TFIDF_COSINE_SIM_FAILED", {"user_email": user_email, "resume_name": resume_file.name, "error": str(e), "traceback": traceback.format_exc()})


            # Determine Predicted Status based on criteria
            predicted_status = "Rejected"
            if similarity_score_percent >= cutoff_score and years_experience >= min_experience:
                predicted_status = "Shortlisted"
            elif years_experience < min_experience:
                predicted_status = "Rejected (Experience)"
            elif similarity_score_percent < cutoff_score:
                predicted_status = "Rejected (Score)"
            
            # Refine prediction based on major skill gaps
            if predicted_status == "Shortlisted" and len(required_skills) > 0 and len(missing_skills) > len(required_skills) / 2: # If too many skills are missing
                 predicted_status = "Rejected (Major Skill Gap)"


            # Generate AI Suggestion based on the *final* predicted status and other factors
            ai_suggestion = generate_concise_ai_suggestion(
                candidate_name=candidate_name,
                score=similarity_score_percent,
                years_exp=years_experience,
                semantic_similarity=0.0 # Placeholder, as semantic_score is called later
            )
            # Re-call semantic_score to get the actual semantic_similarity for the result
            actual_score, _, semantic_similarity_val = semantic_score(resume_text, jd_text, years_experience)
            # Update AI suggestion with actual semantic similarity if needed
            ai_suggestion = generate_concise_ai_suggestion(
                candidate_name=candidate_name,
                score=actual_score,
                years_exp=years_experience,
                semantic_similarity=semantic_similarity_val
            )


            # Match Level based on score
            match_level = ""
            if similarity_score_percent >= 80:
                match_level = "High"
            elif similarity_score_percent >= 60:
                match_level = "Medium"
            else:
                match_level = "Low"

            results.append({
                "Resume Name": resume_file.name,
                "Candidate Name": candidate_name,
                "Email": email or "N/A",
                "Phone": phone or "N/A",
                "Years Experience": years_experience,
                "Score (%)": similarity_score_percent, # Renamed for clarity in email_page.py
                "Matched Skills": ", ".join(matched_skills) if matched_skills else "None",
                "Missing Skills": ", ".join(missing_skills) if missing_skills else "None",
                "Predicted Status": predicted_status,
                "Match Level": match_level,
                "AI Suggestion": ai_suggestion, # This is the concise one for the table
                "Detailed HR Assessment": generate_detailed_hr_assessment(candidate_name, similarity_score_percent, years_experience, semantic_similarity_val, jd_text, resume_text), # Store the detailed one for top candidate
                "Semantic Similarity": semantic_similarity_val,
                "Resume Raw Text": resume_text, # Store full text for potential future use (e.g., detailed view)
                "WordCloudText": clean_text_for_wordcloud(resume_text) # For analytics word cloud
            })
            log_user_action(user_email, "RESUME_PROCESSED", {
                "resume_name": resume_file.name,
                "score": similarity_score_percent,
                "predicted_status": predicted_status,
                "years_exp": years_experience
            })
            update_metrics_summary("total_resumes_screened", 1)
            update_metrics_summary("user_resumes_screened", 1, user_email=user_email)
        
        my_bar.empty()
        status_text.empty() # Clear the status text after processing
        st.success("Screening complete! Check results below.")
        log_user_action(user_email, "SCREENING_COMPLETE_SUCCESS", {"num_processed": len(results), "num_failed_to_parse": len(uploaded_resumes) - len(results)})

        df_results = pd.DataFrame(results)
        st.session_state['screening_results'] = df_results # Store results in session state for other pages

        # Save results to CSV for analytics.py to use (re-added as analytics.py was updated to use it)
        # Ensure 'data' directory exists for results.csv
        if not os.path.exists("data"):
            os.makedirs("data")
        df_results.to_csv(os.path.join("data", "results.csv"), index=False)
        log_system_event("INFO", "SCREENING_RESULTS_SAVED_TO_CSV", {"rows": len(df_results)})


        # --- Overall Candidate Comparison Chart ---
        st.markdown("## 📊 Candidate Score Comparison")
        st.caption("Visual overview of how each candidate ranks against the job requirements.")
        if not df_results.empty:
            try:
                fig, ax = plt.subplots(figsize=(12, 7))
                # Define colors: Green for top, Yellow for moderate, Red for low
                colors = ['#4CAF50' if s >= cutoff_score else '#FFC107' if s >= (cutoff_score * 0.75) else '#F44346' for s in df_results['Score (%)']]
                bars = ax.bar(df_results['Candidate Name'], df_results['Score (%)'], color=colors)
                ax.set_xlabel("Candidate", fontsize=14)
                ax.set_ylabel("Score (%)", fontsize=14)
                ax.set_title("Resume Screening Scores Across Candidates", fontsize=16, fontweight='bold')
                ax.set_ylim(0, 100)
                plt.xticks(rotation=60, ha='right', fontsize=10)
                plt.yticks(fontsize=10)
                for bar in bars:
                    yval = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2, yval + 1, f"{yval:.1f}", ha='center', va='bottom', fontsize=9)
                plt.tight_layout()
                st.pyplot(fig)
                plt.close(fig) # Close the figure to free up memory
            except Exception as e:
                st.error("Error generating Candidate Score Comparison chart.")
                log_system_event("ERROR", "PLOT_GENERATION_FAILED", {"chart": "Candidate Score Comparison", "error": str(e), "traceback": traceback.format_exc()})
        else:
            st.info("Upload resumes to see a comparison chart.")

        st.markdown("---")

        # --- TOP CANDIDATE AI RECOMMENDATION (Game Changer Feature) ---
        st.markdown("## 👑 Top Candidate AI Assessment")
        st.caption("A concise, AI-powered assessment for the most suitable candidate.")
        
        if not df_results.empty:
            # Sort by score descending to ensure top_candidate is truly the highest scored
            df_results_sorted = df_results.sort_values(by='Score (%)', ascending=False).reset_index(drop=True)
            top_candidate = df_results_sorted.iloc[0] 

            st.markdown(f"### **{top_candidate['Candidate Name']}**")
            st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **Semantic Similarity:** {top_candidate['Semantic Similarity']:.2f}")
            st.markdown(f"**AI Assessment:**")
            st.markdown(top_candidate['Detailed HR Assessment']) # Display the detailed HR assessment here
            
            # Action button for the top candidate
            if top_candidate['Email'] != "N/A":
                mailto_link_top = create_mailto_link(
                    recipient_email=top_candidate['Email'],
                    candidate_name=top_candidate['Candidate Name'],
                    job_title=jd_source if jd_source != "Paste Manually" else "Job Opportunity" # Use selected JD name
                )
                st.markdown(f'<a href="{mailto_link_top}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Top Candidate for Interview</button></a>', unsafe_allow_html=True)
                log_user_action(user_email, "TOP_CANDIDATE_EMAIL_LINK_GENERATED", {"candidate_name": top_candidate['Candidate Name']})
            else:
                st.info(f"Email address not found for {top_candidate['Candidate Name']}. Cannot send automated invitation.")
            
            st.markdown("---")
            st.info("For detailed analytics, matched keywords, and missing skills for ALL candidates, please navigate to the **Analytics Dashboard**.")

        else:
            st.info("No candidates processed yet to determine the top candidate.")


        # === AI Recommendation for Shortlisted Candidates (Streamlined) ===
        # This section now focuses on a quick summary for *all* shortlisted,
        # with the top one highlighted above.
        st.markdown("## 🌟 Shortlisted Candidates Overview")
        st.caption("Candidates meeting your score and experience criteria.")

        shortlisted_candidates = df_results[(df_results['Score (%)'] >= cutoff_score) & (df_results['Years Experience'] >= min_experience)]

        if not shortlisted_candidates.empty:
            st.success(f"**{len(shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff_score}%, Experience ≥ {min_experience} years).")
            
            # Display a concise table for shortlisted candidates
            display_shortlisted_summary_cols = [
                'Candidate Name',
                'Score (%)',
                'Years Experience',
                'Semantic Similarity',
                'Email', # Include email here for quick reference
                'AI Suggestion' # This is the concise AI suggestion
            ]
            
            st.dataframe(
                shortlisted_candidates[display_shortlisted_summary_cols],
                use_container_width=True,
                hide_index=True,
                column_config={
                    "Score (%)": st.column_config.ProgressColumn(
                        "Score (%)",
                        help="Matching score against job requirements",
                        format="%f",
                        min_value=0,
                        max_value=100,
                    ),
                    "Years Experience": st.column_config.NumberColumn(
                        "Years Experience",
                        help="Total years of professional experience",
                        format="%.1f years",
                    ),
                    "Semantic Similarity": st.column_config.NumberColumn(
                        "Semantic Similarity",
                        help="Conceptual similarity between JD and Resume (higher is better)",
                        format="%.2f",
                        min_value=0,
                        max_value=1
                    ),
                    "AI Suggestion": st.column_config.Column(
                        "AI Suggestion",
                        help="AI's concise overall assessment and recommendation"
                    )
                }
            )
            st.info("For individual detailed AI assessments and action steps, please refer to the table above or the Analytics Dashboard.")

        else:
            st.warning("No candidates met the defined screening criteria (score cutoff and minimum experience). You might consider adjusting the sliders or reviewing the uploaded resumes/JD.")

        st.markdown("---")

        # Add a 'Tag' column for quick categorization
        df_results['Tag'] = df_results.apply(lambda row: 
            "👑 Exceptional Match" if row['Score (%)'] >= 90 and row['Years Experience'] >= 5 and row['Semantic Similarity'] >= 0.85 else (
            "🔥 Strong Candidate" if row['Score (%)'] >= 80 and row['Years Experience'] >= 3 and row['Semantic Similarity'] >= 0.7 else (
            "✨ Promising Fit" if row['Score (%)'] >= 60 and row['Years Experience'] >= 1 else (
            "⚠️ Needs Review" if row['Score (%)'] >= 40 else 
            "❌ Limited Match"))), axis=1)

        st.markdown("## 📋 Comprehensive Candidate Results Table")
        st.caption("Full details for all processed resumes. **For deep dive analytics and keyword breakdowns, refer to the Analytics Dashboard.**")
        
        # Define columns to display in the comprehensive table
        comprehensive_cols = [
            'Candidate Name',
            'Score (%)',
            'Years Experience',
            'Semantic Similarity',
            'Tag', # Keep the custom tag
            'Email',
            'AI Suggestion', # This will still contain the concise AI suggestion text
            'Matched Skills', # Changed from Matched Keywords to Matched Skills as per previous code
            'Missing Skills',
            # 'Resume Raw Text' # Removed from default display to keep table manageable, can be viewed in Analytics
        ]
        
        # Ensure all columns exist before trying to display them
        final_display_cols = [col for col in comprehensive_cols if col in df_results.columns]

        st.dataframe(
            df_results[final_display_cols],
            use_container_width=True,
            hide_index=True,
            column_config={
                "Score (%)": st.column_config.ProgressColumn(
                    "Score (%)",
                    help="Matching score against job requirements",
                    format="%f",
                    min_value=0,
                    max_value=100,
                ),
                "Years Experience": st.column_config.NumberColumn(
                    "Years Experience",
                    help="Total years of professional experience",
                    format="%.1f years",
                ),
                "Semantic Similarity": st.column_config.NumberColumn(
                    "Semantic Similarity",
                    help="Conceptual similarity between JD and Resume (higher is better)",
                    format="%.2f",
                    min_value=0,
                    max_value=1
                ),
                "AI Suggestion": st.column_config.Column(
                    "AI Suggestion",
                    help="AI's concise overall assessment and recommendation"
                ),
                "Matched Skills": st.column_config.Column(
                    "Matched Skills",
                    help="Keywords found in both JD and Resume"
                ),
                "Missing Skills": st.column_config.Column(
                    "Missing Skills",
                    help="Key skills from JD not found in Resume"
                ),
            }
        )

        st.info("Remember to check the Analytics Dashboard for in-depth visualizations of skill overlaps, gaps, and other metrics!")
    else:
        st.info("Please upload a Job Description and at least one Resume to begin the screening process.")

    st.markdown("</div>", unsafe_allow_html=True)

# This block ensures that if screener.py is run directly, it initializes Streamlit
if __name__ == "__main__":
    st.set_page_config(layout="wide", page_title="Resume Screener")
    st.title("Resume Screener (Standalone Test Mode)")

    # Mock user session state for standalone testing
    if "user_email" not in st.session_state:
        st.session_state.user_email = "test_screener_user@example.com"
        st.info("Running in standalone mode. Mocking user: test_screener_user@example.com")
    
    # Initialize screening_results if not present for standalone run
    if 'screening_results' not in st.session_state:
        st.session_state['screening_results'] = pd.DataFrame()
    
    # Dummy required skills and min_experience for standalone, ensuring consistency with session_state usage
    if 'screening_min_experience' not in st.session_state:
        st.session_state['screening_min_experience'] = 2
    if 'screening_cutoff_score' not in st.session_state:
        st.session_state['screening_cutoff_score'] = 75

    resume_screener_page() # Call the main function for the page
