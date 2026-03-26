# 🫘 Dry Bean Classifier — Streamlit App

Beautiful Agri-Tech ML app for classifying 7 Turkish dry bean varieties using 16 geometric features.

## 📁 Required File Structure

```
your-repo/
├── app.py
├── requirements.txt
├── .streamlit/
│   └── config.toml
└── images/
    ├── BG_IMAGE_opt.jpg        ← Blurred dark forest background
    ├── BARBUNYA.jpeg
    ├── BOMBAY.jpeg
    ├── CALI.jpeg
    ├── DERMASON.jpeg
    ├── HOROZ.jpeg
    ├── SEKER.jpeg
    ├── SIRA.jpeg
    ├── 01_class_distribution.png
    ├── 04_correlation_heatmap.png
    ├── 06_confusion_matrices.png
    ├── 07_overfitting_check.png
    ├── 08_model_comparison_table.png
    └── 09_feature_importance.png
```

## 🚀 Deploy to Streamlit Cloud (Free)

1. **Create a public GitHub repository**
2. **Upload ALL files** maintaining the exact folder structure above
3. Go to **[share.streamlit.io](https://share.streamlit.io)**
4. Click **New App** → select your repo
5. Set **Main file path**: `app.py`
6. Click **Deploy** — live in ~2 minutes!

## 💻 Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

## ✨ App Features

| Tab | Content |
|-----|---------|
| 🔬 **Classify a Bean** | 16 input sliders + number inputs → real-time prediction with bean photo + description |
| 📊 **EDA & Model Analysis** | Class distribution, correlation heatmap, feature importance, model comparison, confusion matrices, overfitting check |
| 🌿 **Bean Encyclopedia** | Visual cards for all 7 varieties with actual photographs, origins, culinary uses, protein info |

## 🫘 Bean Varieties
DERMASON · SIRA · SEKER · HOROZ · CALI · BARBUNYA · BOMBAY

## 🎨 Design
- **Background**: Your forest image blurred dark overlay
- **Font**: Cormorant Garamond (headers) + Outfit (body) + JetBrains Mono (data)
- **Theme**: Deep forest green on dark, glass morphism cards
