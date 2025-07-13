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

# --- Load Embedding + ML Model ---
@st.cache_resource
def load_ml_model():
    try:
        model = SentenceTransformer("all-MiniLM-L6-v2")
        ml_model = joblib.load("ml_screening_model.pkl")
        return model, ml_model
    except Exception as e:
        st.error(f"❌ Error loading models: {e}. Please ensure 'ml_screening_model.pkl' is in the same directory.")
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

# --- Helpers ---
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
            return ''.join(page.extract_text() or '' for page in pdf.pages)
    except Exception as e:
        return f"[ERROR] {str(e)}"

def extract_years_of_experience(text):
    """Extracts years of experience from a given text by parsing date ranges or keywords."""
    text = text.lower()
    total_months = 0
    # Corrected regex: changed '[a-2]*' to '[a-z]*'
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
        match = re.search(r'(\d+(?:\.\d+)?)\s*(\+)?\s*(year|yrs|years)\b', text)
        if not match:
            match = re.search(r'experience[^\d]{0,10}(\d+(?:\.\d+)?)', text)
        if match:
            return float(match.group(1))

    return round(total_months / 12, 1)

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

    if score >= 85 and years_exp >= 4 and semantic_similarity >= 0.75:
        overall_fit_description = "High alignment with job requirements."
        review_focus_text = "Focus on cultural fit and specific project contributions."
    elif score >= 65 and years_exp >= 2 and semantic_similarity >= 0.4:
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

    # Tier 1: Exceptional Candidate
    if score >= 90 and years_exp >= 5 and semantic_similarity >= 0.85:
        overall_assessment_title = "Exceptional Candidate: Highly Aligned with Strategic Needs"
        assessment_parts.append(f"**{candidate_name}** presents an **exceptional profile** with a high score of {score:.2f}% and {years_exp:.1f} years of experience. This demonstrates a profound alignment with the job description's core requirements, further evidenced by a strong semantic similarity of {semantic_similarity:.2f}.")
        assessment_parts.append("This candidate possesses a robust skill set directly matching critical keywords in the JD, suggesting immediate productivity and minimal ramp-up time. Their extensive experience indicates a capacity for leadership and handling complex challenges. They are poised to make significant contributions from day one.")
        next_steps_focus = "The next steps should focus on assessing cultural integration, exploring leadership potential, and delving into strategic contributions during the interview. This candidate appears to be a strong fit for a pivotal role within the organization."
    # Tier 2: Strong Candidate
    elif score >= 80 and years_exp >= 3 and semantic_similarity >= 0.7:
        overall_assessment_title = "Strong Candidate: Excellent Potential for Key Contributions"
        assessment_parts.append(f"**{candidate_name}** is a **strong candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. They show excellent alignment with the job description, supported by a solid semantic similarity of {semantic_similarity:.2f}.")
        assessment_parts.append("Key strengths include a significant overlap in required skills and practical experience that directly addresses the job's demands. This individual is likely to integrate well and contribute effectively from an early stage, bringing valuable expertise to the team.")
        next_steps_focus = "During the interview, explore specific project methodologies, problem-solving approaches, and long-term career aspirations to confirm alignment with team dynamics and growth opportunities within the company."
    # Tier 3: Promising Candidate
    elif score >= 60 and years_exp >= 1 and semantic_similarity >= 0.35:
        overall_assessment_title = "Promising Candidate: Requires Focused Review on Specific Gaps"
        assessment_parts.append(f"**{candidate_name}** is a **promising candidate** with a score of {score:.2f}% and {years_exp:.1f} years of experience. While demonstrating a foundational understanding (semantic similarity: {semantic_similarity:.2f}), there are areas that warrant deeper investigation to ensure a complete fit.")
        
        gaps = []
        if score < 70:
            gaps.append("The overall score suggests some core skill areas may need development or further clarification.")
        if years_exp < 2:
            gaps.append(f"Experience ({years_exp:.1f} yrs) is on the lower side; assess their ability to scale up quickly and take on more responsibility.")
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
        st.warning("ML models not loaded. Providing basic score and generic feedback.")
        # Simplified fallback for score and feedback
        # Use the new extraction logic for fallback as well
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
    # st.set_page_config(layout="wide", page_title="ScreenerPro - AI Resume Screener", page_icon="🧠") # Removed: should be in main.py
    st.title("🧠 ScreenerPro – AI-Powered Resume Screener")

    # --- Job Description and Controls Section ---
    st.markdown("## ⚙️ Define Job Requirements & Screening Criteria")
    col1, col2 = st.columns([2, 1])

    with col1:
        jd_text = ""
        job_roles = {"Upload my own": None}
        if os.path.exists("data"):
            for fname in os.listdir("data"):
                if fname.endswith(".txt"):
                    job_roles[fname.replace(".txt", "").replace("_", " ").title()] = os.path.join("data", fname)

        jd_option = st.selectbox("📌 **Select a Pre-Loaded Job Role or Upload Your Own Job Description**", list(job_roles.keys()))
        if jd_option == "Upload my own":
            jd_file = st.file_uploader("Upload Job Description (TXT)", type="txt", help="Upload a .txt file containing the job description.")
            if jd_file:
                jd_text = jd_file.read().decode("utf-8")
        else:
            jd_path = job_roles[jd_option]
            if jd_path and os.path.exists(jd_path):
                with open(jd_path, "r", encoding="utf-8") as f:
                    jd_text = f.read()
        
        if jd_text:
            with st.expander("📝 View Loaded Job Description"):
                st.text_area("Job Description Content", jd_text, height=200, disabled=True, label_visibility="collapsed")

    with col2:
        # Store cutoff and min_experience in session state
        cutoff = st.slider("📈 **Minimum Score Cutoff (%)**", 0, 100, 75, help="Candidates scoring below this percentage will be flagged for closer review or considered less suitable.")
        st.session_state['screening_cutoff_score'] = cutoff # Store in session state

        min_experience = st.slider("💼 **Minimum Experience Required (Years)**", 0, 15, 2, help="Candidates with less than this experience will be noted.")
        st.session_state['screening_min_experience'] = min_experience # Store in session state

        st.markdown("---")
        st.info("Once criteria are set, upload resumes below to begin screening.")

    resume_files = st.file_uploader("📄 **Upload Resumes (PDF)**", type="pdf", accept_multiple_files=True, help="Upload one or more PDF resumes for screening.")

    df = pd.DataFrame()

    if jd_text and resume_files:
        # --- Job Description Keyword Cloud ---
        st.markdown("---")
        st.markdown("## ☁️ Job Description Keyword Cloud")
        st.caption("Visualizing the most frequent and important keywords from the Job Description.")
        
        # Use the new extract_relevant_keywords function for the word cloud
        # Pass MASTER_SKILLS directly as the filter_set
        jd_words_for_cloud_set = extract_relevant_keywords(jd_text, MASTER_SKILLS)

        jd_words_for_cloud = " ".join(list(jd_words_for_cloud_set))

        if jd_words_for_cloud:
            wordcloud = WordCloud(width=800, height=400, background_color='white', collocations=False).generate(jd_words_for_cloud)
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.imshow(wordcloud, interpolation='bilinear')
            ax.axis('off')
            st.pyplot(fig)
            plt.close(fig) # Close the figure to free up memory
        else:
            st.info("No significant keywords to display for the Job Description. Please ensure your JD has sufficient content or adjust your MASTER_SKILLS list.")
        st.markdown("---")

        results = []
        resume_text_map = {}
        progress_bar = st.progress(0)
        status_text = st.empty()

        for i, file in enumerate(resume_files):
            status_text.text(f"Processing {file.name} ({i+1}/{len(resume_files)})...")
            progress_bar.progress((i + 1) / len(resume_files))

            text = extract_text_from_pdf(file)
            if text.startswith("[ERROR]"):
                st.error(f"Failed to process {file.name}: {text.replace('[ERROR] ', '')}")
                continue

            exp = extract_years_of_experience(text)
            email = extract_email(text)
            candidate_name = extract_name(text) or file.name.replace('.pdf', '').replace('_', ' ').title()

            # Calculate Matched Keywords and Missing Skills using the new function
            resume_words_set = extract_relevant_keywords(text, MASTER_SKILLS)
            jd_words_set = extract_relevant_keywords(jd_text, MASTER_SKILLS)

            matched_keywords = list(resume_words_set.intersection(jd_words_set))
            missing_skills = list(jd_words_set.difference(resume_words_set)) 

            # semantic_score now returns score, placeholder feedback, semantic_similarity
            score, _, semantic_similarity = semantic_score(text, jd_text, exp)
            
            # Generate the CONCISE AI suggestion for the table
            concise_ai_suggestion = generate_concise_ai_suggestion(
                candidate_name=candidate_name,
                score=score,
                years_exp=exp,
                semantic_similarity=semantic_similarity
            )

            # Generate the DETAILED HR assessment for the top candidate section
            detailed_hr_assessment = generate_detailed_hr_assessment(
                candidate_name=candidate_name,
                score=score,
                years_exp=exp,
                semantic_similarity=semantic_similarity,
                jd_text=jd_text,
                resume_text=text
            )

            results.append({
                "File Name": file.name,
                "Candidate Name": candidate_name,
                "Score (%)": score,
                "Years Experience": exp,
                "Email": email or "Not Found",
                "AI Suggestion": concise_ai_suggestion, # This is the concise one for the table
                "Detailed HR Assessment": detailed_hr_assessment, # Store the detailed one for top candidate
                "Matched Keywords": ", ".join(matched_keywords), # Added Matched Keywords
                "Missing Skills": ", ".join(missing_skills),    # Added Missing Skills
                "Semantic Similarity": semantic_similarity,
                "Resume Raw Text": text
            })
            resume_text_map[file.name] = text
        
        progress_bar.empty()
        status_text.empty()


        df = pd.DataFrame(results).sort_values(by="Score (%)", ascending=False).reset_index(drop=True)

        st.session_state['screening_results'] = results
        
        # Save results to CSV for analytics.py to use (re-added as analytics.py was updated to use it)
        df.to_csv("results.csv", index=False)


        # --- Overall Candidate Comparison Chart ---
        st.markdown("## 📊 Candidate Score Comparison")
        st.caption("Visual overview of how each candidate ranks against the job requirements.")
        if not df.empty:
            fig, ax = plt.subplots(figsize=(12, 7))
            # Define colors: Green for top, Yellow for moderate, Red for low
            colors = ['#4CAF50' if s >= cutoff else '#FFC107' if s >= (cutoff * 0.75) else '#F44346' for s in df['Score (%)']]
            bars = ax.bar(df['Candidate Name'], df['Score (%)'], color=colors)
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
        else:
            st.info("Upload resumes to see a comparison chart.")

        st.markdown("---")

        # --- TOP CANDIDATE AI RECOMMENDATION (Game Changer Feature) ---
        st.markdown("## 👑 Top Candidate AI Assessment")
        st.caption("A concise, AI-powered assessment for the most suitable candidate.")
        
        if not df.empty:
            top_candidate = df.iloc[0] # Get the top candidate (already sorted by score)
            st.markdown(f"### **{top_candidate['Candidate Name']}**")
            st.markdown(f"**Score:** {top_candidate['Score (%)']:.2f}% | **Experience:** {top_candidate['Years Experience']:.1f} years | **Semantic Similarity:** {top_candidate['Semantic Similarity']:.2f}")
            st.markdown(f"**AI Assessment:**")
            st.markdown(top_candidate['Detailed HR Assessment']) # Display the detailed HR assessment here
            
            # Action button for the top candidate
            if top_candidate['Email'] != "Not Found":
                mailto_link_top = create_mailto_link(
                    recipient_email=top_candidate['Email'],
                    candidate_name=top_candidate['Candidate Name'],
                    job_title=jd_option if jd_option != "Upload my own" else "Job Opportunity"
                )
                st.markdown(f'<a href="{mailto_link_top}" target="_blank"><button style="background-color:#00cec9;color:white;border:none;padding:10px 20px;text-align:center;text-decoration:none;display:inline-block;font-size:16px;margin:4px 2px;cursor:pointer;border-radius:8px;">📧 Invite Top Candidate for Interview</button></a>', unsafe_allow_html=True)
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

        shortlisted_candidates = df[(df['Score (%)'] >= cutoff) & (df['Years Experience'] >= min_experience)]

        if not shortlisted_candidates.empty:
            st.success(f"**{len(shortlisted_candidates)}** candidate(s) meet your specified criteria (Score ≥ {cutoff}%, Experience ≥ {min_experience} years).")
            
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
        df['Tag'] = df.apply(lambda row: 
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
            'Matched Keywords',
            'Missing Skills',
            # 'Resume Raw Text' # Removed from default display to keep table manageable, can be viewed in Analytics
        ]
        
        # Ensure all columns exist before trying to display them
        final_display_cols = [col for col in comprehensive_cols if col in df.columns]

        st.dataframe(
            df[final_display_cols],
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
                "Matched Keywords": st.column_config.Column(
                    "Matched Keywords",
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
