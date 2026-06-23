# 🏦 BSF AI Banking Intelligence System

<div align="center">

![BSF AI System](https://img.shields.io/badge/BSF-AI%20Banking%20Intelligence-3D3DDB?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PHBhdGggZmlsbD0id2hpdGUiIGQ9Ik0xMiAyTDIgN2wxMCA1IDEwLTV6TTIgMTdsOCA0IDgtNE0yIDEybDggNCA4LTQiLz48L3N2Zz4=)
![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LightGBM](https://img.shields.io/badge/LightGBM-ML%20Models-2E7D32?style=for-the-badge)
![License](https://img.shields.io/badge/License-Proprietary-red?style=for-the-badge)

**An end-to-end AI-powered banking intelligence platform built for BSF (Saudi French Bank)**  
*Developed by Accord Business Group (ABG)*

[🌐 Live Demo](https://bsf-ai-system-pnvca6wxqlt2ykqmpyznhu.streamlit.app) · [📧 Contact ABG](mailto:info@accordbg.com)

</div>

---

## 📋 Overview

The **BSF AI Banking Intelligence System** is a production-ready, multi-module AI platform that transforms raw customer data into actionable banking intelligence. Built with Streamlit and powered by machine learning models, it provides real-time customer segmentation, next-best-action recommendations, behavioral analysis, and credit decisioning.

---

## ✨ Key Features

| Module | Description | Technology |
|--------|-------------|------------|
| 👥 **Segmentation** | K-Means clustering of 5,000+ customers into 5 behavioral segments | K-Means · RobustScaler · PCA |
| 🎯 **NBA Engine** | AI-powered Next-Best-Action cross-sell recommendations | LightGBM · 5 Binary Classifiers |
| 📊 **Behavior Analysis** | Clickstream & transaction behavioral scoring | IsolationForest · Z-Score |
| 💳 **Credit Decision** | Automated credit scoring and loan eligibility | XGBoost · Random Forest · Gradient Boosting |
| 🔄 **Full Pipeline** | End-to-end 12-step AI pipeline visualization | Interactive Dashboard |
| 🤖 **Live Demo** | Real-time customer profiling and recommendations | Full Pipeline Integration |

---

## 🏗️ System Architecture

```
Raw Data (5K Customers, 300K Transactions)
         │
         ▼
┌─────────────────────────────────────────────┐
│           Feature Engineering               │
│  23 features · RobustScaler normalization   │
└─────────────────────────────────────────────┘
         │
         ▼
┌──────────────────┐    ┌─────────────────────┐
│   Segmentation   │    │    NBA Engine        │
│  K-Means (K=5)   │───▶│  LightGBM × 5       │
│  Standard→VIP    │    │  AUC: 0.79–0.97     │
└──────────────────┘    └─────────────────────┘
         │                        │
         ▼                        ▼
┌──────────────────┐    ┌─────────────────────┐
│ Behavior Module  │    │   Credit Decision   │
│ IsoForest+ZScore │    │  XGB+RF+GBM Ensemble│
│ Clickstream Data │    │  Limit+Rate+Loan    │
└──────────────────┘    └─────────────────────┘
```

---

## 📊 ML Model Performance

### NBA Cross-sell Models (LightGBM)

| Product | ROC-AUC | F1 Score | Avg Precision |
|---------|---------|----------|---------------|
| ✈️ World Travel Card | **0.968** | 0.876 | 0.957 |
| 🏠 Home Finance | **0.906** | 0.839 | 0.935 |
| 💳 Cashback Card | **0.860** | 0.819 | 0.945 |
| 💰 Personal Finance | **0.790** | 0.813 | 0.959 |
| 📈 Savings Account | **0.556** | 0.485 | 0.417 |

### Customer Segmentation

| Segment | Customers | Avg Income | Coverage |
|---------|-----------|------------|----------|
| 👑 VIP | 410 (8.2%) | SAR 48,000 | 88% |
| 💎 Platinum | 760 (15.2%) | SAR 24,000 | 76% |
| 🥇 Gold | 1,100 (22%) | SAR 14,000 | 72% |
| 🥈 Silver | 1,250 (25%) | SAR 8,500 | 54% |
| 👤 Standard | 1,480 (29.6%) | SAR 4,800 | 31% |

---

## 🚀 Getting Started

### Prerequisites

```bash
Python >= 3.11
Git
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/MSABER-ABG/BSF-AI-System.git
cd BSF-AI-System

# 2. Navigate to the app directory
cd bsf_v25

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the application
streamlit run app.py
```

### Or use the Windows launcher

```bash
run.bat
```

---

## 📁 Project Structure

```
BSF-AI-System/
└── bsf_v25/
    ├── app.py                  # Main application entry point
    ├── requirements.txt        # Python dependencies
    ├── run.bat                 # Windows launcher
    │
    ├── pages/
    │   ├── overview.py         # System overview dashboard
    │   ├── pipeline.py         # 12-step AI pipeline visualization
    │   ├── module1.py          # Segmentation & NBA Engine
    │   ├── segmentation.py     # Deep segmentation analysis
    │   ├── module2.py          # Behavioral analysis
    │   ├── module3.py          # Credit decision engine
    │   └── demo.py             # Live interactive demo
    │
    └── utils/
        ├── engine.py           # Core business logic & rules
        ├── styling.py          # ABG brand theme & UI components
        ├── ml_nba.py           # LightGBM NBA models
        └── segmentation_viz.py # K-Means & PCA computations
```

---

## 🛠️ Tech Stack

```python
Frontend    : Streamlit 1.32+
ML Models   : LightGBM, XGBoost, Scikit-learn
Visualization: Plotly 5.18+
Data        : Pandas, NumPy
Deployment  : Streamlit Community Cloud
Version Control: Git + GitHub
```

---

## 📈 Business Impact

```
📊 Revenue Uplift        +9.2%   (projected, 30 days)
🛡️ Churn Prevention     1,240   customers
💰 Retained Spend        2.7B SAR
🎯 NBA Acceptance Rate   31.5%
👥 Segment Engagement    62.8%
```

---

## 🔒 Data & Privacy

- All customer data used in this system is **synthetic** and generated programmatically
- No real BSF customer PII is stored or processed
- Synthetic data is generated using Saudi demographic distributions to reflect realistic banking patterns

---

## 👨‍💼 Development Team

| Role | Organization |
|------|-------------|
| **AI/ML Architecture & Development** | Accord Business Group (ABG) |
| **Business Domain Expertise** | BSF — Saudi French Bank |
| **Data Science Lead** | Dr. M. Saber, PhD — ITMO University |

---

## 📄 License

This project is proprietary software developed exclusively for **BSF (Saudi French Bank)** by **Accord Business Group (ABG)**.  
Unauthorized reproduction or distribution is strictly prohibited.

---

<div align="center">

**Built with ❤️ by Accord Business Group**  
*Transforming Banking Intelligence with AI*

</div>
