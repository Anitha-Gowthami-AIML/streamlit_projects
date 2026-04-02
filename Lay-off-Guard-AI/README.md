# ⚡ LayoffGuard AI — Streamlit Layoff Risk Predictor

> **AI-Powered Career Risk Intelligence Platform**  
> 6 ML Models + ANN Ensemble · 40+ Real-World Variables · Personalized Upskilling Roadmap

---

## 🖥️ Screenshots

- **Prediction Tab** → Risk gauge, emoji pop-up, per-model breakdown, radar chart, action plan  
- **Model Analytics** → ROC curves, confusion matrices, comparison bar charts, ANN training history  
- **Feature Insights** → Feature importance, correlation heatmap, top-20 table  
- **Dataset Overview** → Stats, distribution plots, sample data, descriptive stats  

---

## 🚀 Quick Start

### 1. Clone / Download
```bash
# Place all files in a folder, e.g. layoff-risk/
cd layoff-risk/
```

### 2. Create Virtual Environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
streamlit run app.py
```

Your browser opens at **http://localhost:8501** automatically.

> ⏳ **First run takes ~2 minutes** to generate 6,000 synthetic records and train all 6 models.  
> After that, everything is **cached** — instant restarts!

---

## 📦 Files
```
layoff-risk/
├── app.py              ← Main Streamlit application (everything in one file)
├── requirements.txt    ← Python dependencies
└── README.md           ← This file
```

---

## 🤖 Models Trained

| Model | Type | Notes |
|---|---|---|
| **XGBoost** | Gradient Boosting | 400 trees, depth 6 |
| **LightGBM** | Gradient Boosting | 400 trees, early stopping |
| **Random Forest** | Ensemble | 250 trees, balanced |
| **Gradient Boosting** | Boosting | 200 estimators |
| **Logistic Regression** | Linear | L2 regularization |
| **ANN** | Deep Neural Net | 5 layers, BatchNorm, Dropout |
| **⭐ Ensemble** | Weighted Average | Best 5 combined |

---

## 📊 40+ Features Used

**Profile** · Age · Job Level · Department · Employment Type · Tenure  
**Performance** · Rating · KPI % · Trend · Projects Won/Failed  
**Skills** · AI/ML · Cloud · Certifications · Upskilling Hours · Automation Risk  
**Compensation** · Salary vs Market · Bonus · Equity Vesting  
**Company** · Revenue Growth · Budget Change · Layoff History · Funding Stage · Size  
**Behavior** · Engagement · Absenteeism · Overtime · Remote Work · Manager Score  
**Networking** · LinkedIn · Awards · Mentorship · Cross-Dept Projects  
**Macro** · Economic Stress · Sector Layoff Rate · Job Demand  

---

## 🎨 UI Features

- 🌆 **Blurred office background** via CSS + Unsplash  
- 🎉 **Animated emoji pop-ups** on prediction (varies by risk level)  
- 🕸️ **Radar chart** of your top risk drivers  
- 🎯 **Gauge chart** showing probability needle  
- 📊 **4-tab layout**: Prediction · Analytics · Features · Dataset  
- 🌙 **Full dark glassmorphism** theme  

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push your code to a **GitHub repo**
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect repo → select `app.py` → **Deploy**

> Add `requirements.txt` to root — Streamlit Cloud auto-installs everything.

---

## ⚠️ Disclaimer
This app uses **synthetically generated data** for demonstration purposes.  
Not intended as real financial, HR, or career advice.
