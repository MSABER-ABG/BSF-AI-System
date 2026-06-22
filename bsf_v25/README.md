# 🏦 BSF AI Banking Intelligence System
### Developed by Accord Business Group (ABG) · June 2025

---

## 📋 Project Structure

```
bsf_streamlit/
├── app.py                      ← Main entry point
├── requirements.txt            ← Dependencies
├── run.bat                     ← Windows one-click launcher
├── README.md
├── pages/
│   ├── overview.py             ← Overview & KPI dashboard
│   ├── pipeline.py             ← Full AI pipeline view
│   ├── module1.py              ← Segmentation & NBA
│   ├── module2.py              ← Behavior Analysis
│   ├── module3.py              ← Credit Decision Engine
│   └── demo.py                 ← Live Customer Demo ⭐
└── utils/
    ├── engine.py               ← All AI logic (core)
    └── styling.py              ← ABG theme & components
```

---

## 🚀 Setup & Run

### Option 1: Windows (Double-click)
```
run.bat
```

### Option 2: VS Code Terminal
```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the app
streamlit run app.py

# 3. Open in browser
http://localhost:8501
```

### Option 3: VS Code Launch Config
Add to `.vscode/launch.json`:
```json
{
  "version": "0.2.0",
  "configurations": [
    {
      "name": "BSF AI Streamlit",
      "type": "python",
      "request": "launch",
      "module": "streamlit",
      "args": ["run", "app.py"],
      "console": "integratedTerminal"
    }
  ]
}
```

---

## 📦 Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| streamlit | ≥1.32 | Web framework |
| plotly | ≥5.18 | Interactive charts |
| pandas | ≥2.0 | Data handling |
| numpy | ≥1.24 | Numerical computing |
| scikit-learn | ≥1.3 | ML models |
| xgboost | ≥2.0 | Gradient boosting |
| scipy | ≥1.11 | Z-Score detection |

---

## 🎯 Pages

| Page | Description |
|------|-------------|
| 🏠 Overview | KPIs + segment/behavior/risk charts + module cards |
| 🔄 Pipeline | Full AI pipeline steps with click-to-detail |
| 👥 Module 1 | Segmentation + NBA formula + segment profiles |
| 📊 Module 2 | Behavior analysis + anomaly + drift detection |
| 💳 Module 3 | Ensemble model + credit rules + human-in-loop |
| 🎯 Live Demo | **Enter any customer → full pipeline → results** |

---

## 🔗 Connecting to Google Colab Outputs

To use real model outputs from the notebooks, replace the engine logic in `utils/engine.py`:

```python
# Load real trained models
import joblib
kmeans = joblib.load('path/to/kmeans_segmentation.pkl')
scaler = joblib.load('path/to/feature_scaler.pkl')
ensemble = joblib.load('path/to/ensemble_risk_model.pkl')

# Load real customer data
df = pd.read_csv('path/to/module1_final_output.csv')
```

---

## 🎨 Design

- **Style:** ABG theme (matches BankGuard AI project)
- **Font:** Plus Jakarta Sans
- **Primary Color:** `#3D3DDB` (ABG Blue)
- **Background:** `#F0F0F6` (Light grey)
- **Sidebar:** White with ABG branding
