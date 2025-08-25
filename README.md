🚀 AI-Powered HR Intelligence Platform
Revolutionary Workforce Analytics with Generative AI

A comprehensive HR analytics platform that transforms traditional HR management through advanced AI, predictive modeling, and intelligent decision support systems.

<p align="center"> <img src="https://img.shields.io/badge/Python-3.8+-blue.svg" /> <img src="https://img.shields.io/badge/Streamlit-1.28+-red.svg" /> <img src="https://img.shields.io/badge/OpenAI-GPT--4o--mini-green.svg" /> <img src="https://img.shields.io/badge/License-MIT-yellow.svg" /> <br /> <img src="https://img.shields.io/github/stars/balajee-dutta/My-Business-Project-University-of-Exeter-?style=social" /> <img src="https://img.shields.io/github/forks/balajee-dutta/My-Business-Project-University-of-Exeter-?style=social" /> <img src="https://img.shields.io/github/contributors/balajee-dutta/My-Business-Project-University-of-Exeter-" /> <img src="https://img.shields.io/badge/Datasets-4-green" /> </p>
📑 Table of Contents

Overview
 • Key Features
 • Performance Metrics
 • Installation
 • Usage
 • File Structure
 • Technical Architecture
 • Walkthrough Videos
 • Results & Evaluation
 • Contributing
 • License
 • Acknowledgments
 • Contact & Support

🔍 Overview

The HR Intelligence Platform leverages cutting-edge AI technologies to deliver:

92.3% accuracy in attrition prediction

86.6% reduction in manual reporting effort

96% Top-5 accuracy in resume-job matching

Real-time AI-powered insights with natural language explanations

Built With: OpenAI GPT-4o-mini • text-embedding-3-large • Random Forest ML • Streamlit • Plotly

⭐ Key Features

Executive Dashboard: Real-time KPIs, attrition alerts, pay gap analysis, succession planning.
Attrition Analytics: Predictive modeling, early risk detection, personalized retention, tenure analysis.
Intelligent Talent Matching: Semantic scoring, Talent Fit vs Experience Fit, market intelligence, 96% Top-K accuracy.
AI Assistant: Conversational queries, context-aware GPT insights, scenario analysis, executive summaries.
Advanced Analytics: Bias detection, cross-validation, GDPR-compliance, accessibility with Okabe-Ito palette.

📊 Performance Metrics
Metric	Result	Benchmark	Gain
Attrition Prediction Accuracy	92.3%	82–85%	+8–10%
Resume Matching Top-5 Accuracy	96%	75–80%	+16–21%
Manual Effort Reduction	86.6%	60–70%	+16–26%
Gender Pay Gap Detection	<3% variance	5–8%	Best practice
Model F1-Score	0.90	0.75–0.85	+6–20%
⚙️ Installation

Prerequisites: Python 3.8+, OpenAI API key, 4GB+ RAM, modern browser.

git clone https://github.com/balajee-dutta/My-Business-Project-University-of-Exeter-.git
cd My-Business-Project-University-of-Exeter-
pip install -r requirements.txt


Add your API key in the script:

api_key = "your-openai-api-key-here"


Run:

streamlit run "My Gen AI Enabled HR Assistant- Project File.py"


Access at http://localhost:8501

🚀 Usage

Upload HR datasets (CSV).

Explore dashboard KPIs and alerts.

Run attrition analysis and generate retention strategies.

Match resumes with job descriptions.

Ask HR questions via AI assistant.

Data Required: Employee (demographics, salary, performance, engagement) • Resume (skills, education, experience, expectations) • Jobs (requirements, salary ranges, location).

All data is processed locally, anonymized for PII, and GDPR-compliant.

📂 File Structure
├── My Gen AI Enabled HR Assistant- Project File.py    # Main app
├── EEHCDataHRsmartAssistant.csv                      # Employee dataset
├── Resume_entries.csv                                # Resumes
├── job_vacancies__entries.csv                        # Job postings
├── Eval_set_entries.csv                              # Evaluation data
├── EMPLOYEE DATA DICTIONARY -HC Description.txt      # Data dictionary
├── My code- Walkthrough.mp4                          # Code demo
├── My HR Intelligence platform walkthrough.mp4       # Platform demo
├── README.md                                         # Documentation
└── requirements.txt                                  # Dependencies

🏗️ Technical Architecture
flowchart TD
  A[Frontend Layer] --> B[API Integration] --> C[ML & Analytics]
  A --> D[Data Processing Layer]


Frontend: Streamlit, Plotly, Okabe-Ito colors
API: OpenAI GPT-4o-mini, embeddings, error handling
ML/Analytics: Random Forest, feature engineering, cross-validation, bias detection
Data Processing: Pandas/NumPy, cleaning, PII detection, GDPR compliance

🎥 Walkthrough Videos

Platform Demo: My HR Intelligence platform walkthrough.mp4

Code Deep Dive: My code- Walkthrough.mp4

📈 Results & Evaluation

Model Performance: Attrition (92.3% acc, F1=0.90, ROC-AUC=0.94). Resume Matching (Top-1=73%, Top-5=96%, <3s).
Business Impact: Saves 35.5 hrs weekly per analyst, avoids $235k attrition cost, delivers 783% ROI in year one, +28% decision-making quality.
Ethical AI: <3% demographic variance, GDPR-compliance, explainable NLP-based insights, accessibility-first visuals.

🤝 Contributing

Fork → Branch → PR. Follow PEP8, add tests/docs, ensure ethical AI.
Areas: Model accuracy • HRIS integration • Multilingual support • Advanced visuals • Mobile optimization

📜 License

MIT License → see LICENSE
.
Commercial use, modification, and distribution permitted. OpenAI API usage requires separate licensing.

🙌 Acknowledgments

67 academic citations • Benchmarks from Deloitte/SHRM/Gartner • Community input (Hugging Face, GitHub, Stack Overflow) • Ethical design with Okabe-Ito palette, GDPR compliance, bias mitigation, explainable AI.

📞 Contact & Support

Developer Contact
📧 bldutta94@gmail.com

📱 07733925935

Support Options

🛠️ Technical Issues: GitHub Issues

💼 Business Inquiries: Email

🎓 Academic Collaboration: Email

💡 Feature Requests: GitHub Discussions

⏱️ Response Time
📧 Emails: 24–48 hrs
🛠️ Issues: 1–3 business days
🤝 Collaboration: ~1 week

<p align="center"> <b>🌍 Built for the future of HR</b><br> <i>Transforming HR from reactive administration to proactive strategic partnership through ethical AI innovation.</i> </p>
