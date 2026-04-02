# 🫘 Dry Bean Classifier — 7 Varieties

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bean-classifier-app-7-varieties.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![SVM Accuracy](https://img.shields.io/badge/SVM%20Accuracy-92.4%25-brightgreen.svg)
![Models](https://img.shields.io/badge/Models%20Trained-9-orange.svg)
![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)

> AI-powered identification of **7 Turkish dry bean varieties** using **16 geometric morphological features** captured via computer vision — built and deployed as part of the **AI & ML Program at IIT Guwahati – IOT Academy**.

---

## 🔗 Live Demo

👉 **[Try the App Here](https://bean-classifier-app-7-varieties.streamlit.app/)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [App Features](#app-features)
- [Bean Varieties & Class Distribution](#bean-varieties--class-distribution)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [ML Pipeline](#ml-pipeline)
- [Model Results](#model-results)
- [Key Insights](#key-insights)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 📖 Overview

Identifying dry bean varieties manually is time-consuming and error-prone in agricultural settings. This project builds an **end-to-end supervised ML classification system** that identifies dry beans into one of 7 registered Turkish varieties using only their **physical shape measurements**.

The dataset originates from a real computer vision study at Selçuk University, Turkey, where **13,611 bean grain images** were photographed with a high-resolution camera and **16 morphological features** were extracted per grain using image segmentation.

The best performing model — **SVM with 92.4% accuracy and zero overfitting** — is deployed as an interactive Streamlit web application with 3 dedicated sections.

---

## ✨ App Features

The app has **3 tabs**:

### 🫘 Tab 1 — Classify a Bean
- Enter all 16 morphological measurements via interactive sliders and input fields
- Grouped into **Size Measurements**, **Shape & Form**, and **Shape Factors**
- Click **"Identify Bean Variety"** to get real-time classification
- Result displays predicted variety name, actual bean photograph, and confidence score

### 📊 Tab 2 — EDA & Model Analysis
Full exploratory data analysis pipeline including:
1. Class distribution bar chart across all 7 varieties
2. Feature correlation heatmap (Pearson, all 16 features)
3. Feature importances chart (Random Forest — top predictors ranked)
4. Model comparison table (all 9 models — Train Acc, Test Acc, F1, CV Mean, Overfitting flag)
5. Overfitting analysis — Train vs Test accuracy chart
6. Confusion matrices for top 3 models (SVM, Logistic Regression, Random Forest)
7. ML pipeline summary (all 8 preprocessing + training steps)

### 🌿 Tab 3 — Bean Encyclopedia
- Visual guide to all 7 varieties with actual bean photographs
- Origins, culinary uses, nutritional data, and classification characteristics per variety

---

## 🫘 Bean Varieties & Class Distribution

| Variety | Samples | % of Dataset | Description |
|---|---|---|---|
| **Dermason** | 3,546 | 26.1% | Smallest bean — Tiny Titan of Turkish Cuisine |
| **Sira** | 2,636 | 19.4% | Elongated All-Rounder |
| **Seker** | 2,027 | 14.9% | Sugar Bean — Sweet & Smooth |
| **Horoz** | 1,928 | 14.2% | Rooster Bean — Bold & Hearty |
| **Cali** | 1,630 | 12.0% | Medium-large oval bean |
| **Barbunya** | 1,322 | 9.7% | Speckled Cranberry Bean |
| **Bombay** | 522 | 3.8% | Largest bean — minority class (handled with SMOTE) |
| **Total** | **13,611** | **100%** | |

> ⚠️ **Class Imbalance:** Bombay (522 samples) vs Dermason (3,546 samples) — handled using **SMOTE + class weighting**.

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository — Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| **Total Samples** | 13,611 |
| **Features** | 16 morphological features |
| **Target Classes** | 7 (Dermason, Sira, Seker, Horoz, Cali, Barbunya, Bombay) |
| **Missing Values** | 0 |
| **License** | CC BY 4.0 |

> **Citation:** Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.

---

## 🔬 Features Used

All 16 morphological features extracted via computer vision:

| # | Feature | Description |
|---|---|---|
| 1 | **Area** | Total pixel count inside bean boundary |
| 2 | **Perimeter** | Length of the bean's outer border |
| 3 | **Major Axis Length** | Longest axis distance |
| 4 | **Minor Axis Length** | Shortest axis distance |
| 5 | **Aspect Ratio** | Major ÷ Minor axis (1 = perfect circle) |
| 6 | **Eccentricity** | 0 = circle, 1 = straight line |
| 7 | **Convex Area** | Smallest convex polygon area |
| 8 | **Equiv. Diameter** | Diameter of circle with same area |
| 9 | **Extent** | Bean area ÷ bounding box area |
| 10 | **Solidity** | Bean area ÷ convex hull area |
| 11 | **Roundness** | 4π·Area ÷ Perimeter² |
| 12 | **Compactness** | EquivDiameter ÷ MajorAxisLength |
| 13 | **ShapeFactor1** ⭐ | MajorAxis ÷ Area — key feature |
| 14 | **ShapeFactor2** | MinorAxis ÷ Area |
| 15 | **ShapeFactor3** ⭐ | Compactness² — **most important** (importance: 0.112) |
| 16 | **ShapeFactor4** | Convexity measure |

> 💡 **Multicollinearity note:** Area ↔ ConvexArea correlation ≈ 1.00. AspectRatio ↔ Compactness = -0.99. Tree-based models handle this natively; linear models benefit from PCA or feature dropping.

---

## 🤖 ML Pipeline

```
Raw Dataset (13,611 × 16 features, 0 missing values)
   │
   ▼
1. Data Loading & EDA
   │  └── Duplicate removal, class distribution analysis
   │
   ▼
2. Outlier Treatment
   │  └── IQR capping applied to all numeric features (capped, not removed)
   │
   ▼
3. Skewness Treatment
   │  └── Box-Cox / Log1p / Yeo-Johnson transformations based on value range
   │
   ▼
4. Feature Scaling
   │  └── StandardScaler applied after outlier treatment (for SVM & LR compatibility)
   │
   ▼
5. Train / Test Split
   │  └── 80/20 stratified split — preserving class proportions in both sets
   │
   ▼
6. Class Imbalance Handling
   │  └── SMOTE + class weighting for BOMBAY minority class (522 samples)
   │
   ▼
7. Model Training — 9 Algorithms
   │  └── SVM | LR | DT | RF | KNN | GBM | AdaBoost | Bagging | Naive Bayes
   │
   ▼
8. Hyperparameter Tuning
   │  └── GridSearchCV / RandomizedSearchCV on top-3 models
   │
   ▼
Best Model (SVM) Saved with Pickle / Joblib
   │
   ▼
Streamlit App Deployed ✅
```

---

## 📊 Model Results

### All Models Comparison

| Model | Train Accuracy | Test Accuracy | F1 Score | CV Mean | Overfitting |
|---|---|---|---|---|---|
| **SVM** 🏆 | 0.9238 | **0.9239** | **0.9238** | 0.9238 | ✅ No |
| Logistic Regression | 0.9234 | 0.9207 | 0.9206 | 0.9226 | ✅ No |
| Random Forest | 1.0000 | 0.9175 | 0.9171 | 0.9203 | ❌ Yes |
| KNN | 0.9361 | 0.9147 | 0.9145 | 0.9192 | ✅ No |
| Decision Tree | 1.0000 | 0.8979 | 0.8976 | 0.8925 | ❌ Yes |
| Naive Bayes | 0.8926 | 0.8947 | 0.8946 | 0.8927 | ✅ No |
| AdaBoost | 0.8701 | 0.8675 | 0.8670 | 0.8788 | ✅ No |

> 🏆 **SVM wins** on all fronts: highest test accuracy (92.4%), best F1 (0.9238), and **zero overfitting gap**. Decision Tree and Random Forest achieved 100% training accuracy — a classic memorization signal.

### Confusion Matrix Highlights (SVM)

- ✅ **Dermason**: 660 correct classifications
- ✅ **Sira**: 452 correct classifications
- ✅ **Bombay**: Near-perfect recall (only 2 errors) despite severe class imbalance
- ⚠️ **Hardest pair**: SIRA ↔ DERMASON — ~58 misclassifications due to shape similarity

### Top Feature Importances (Random Forest)

| Rank | Feature | Importance |
|---|---|---|
| 1 | ShapeFactor3 | 0.112 |
| 2 | ShapeFactor1 | 0.094 |
| 3 | Perimeter | 0.093 |
| 4 | MajorAxisLength | 0.089 |

> 💡 Shape-based metrics outperform raw size measurements for classification.

---

## 💡 Key Insights

- 🏆 **SVM is the best model** — 92.4% test accuracy with zero overfitting, confirming excellent generalisation
- 🌿 **Shape factors dominate** — ShapeFactor3 (0.112) is the single most predictive feature, not raw size
- 🐘 **Bombay is the easiest to classify** — only 2 errors despite being the smallest class (522 samples)
- 🔀 **Dermason ↔ Sira is the hardest pair** — ~58 SVM misclassifications due to overlapping shape profiles
- ⚠️ **Random Forest and Decision Tree overfit** — both hit 100% training accuracy, confirming memorisation
- 📊 **Class imbalance is real** — Dermason (3,546) is 6.8× larger than Bombay (522), handled with SMOTE

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, cleaning & manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib & Seaborn** | EDA visualizations — heatmaps, bar charts, confusion matrices |
| **Scikit-learn** | All ML models, preprocessing, evaluation, GridSearchCV |
| **Pickle / Joblib** | Saving & loading trained models |
| **Streamlit** | Interactive 3-tab web app & cloud deployment |

---

## 📁 Project Structure

```
Bean_Classification_Model/
│
├── app.py                        # Main Streamlit application (3 tabs)
├── bean_model.pkl                # Trained SVM model (TODO: confirm filename)
├── Dry_Bean_Dataset.xlsx         # UCI dataset (TODO: confirm filename)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

> 📌 **TODO: Confirm your actual `.pkl` model filename and dataset filename**

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Anitha-Gowthami-AIML/streamlit_projects.git
cd streamlit_projects/Bean_Classification_Model
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the App
```bash
streamlit run app.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## 📸 Screenshots

| Landing Page | Input Form |
|---|---|
| ![Landing](screenshots/landing.png) | ![Input](screenshots/input.png) |

| Model Comparison | Confusion Matrices |
|---|---|
| ![Models](screenshots/model_comparison.png) | ![CM](screenshots/confusion_matrices.png) |

| Feature Importances | Bean Encyclopedia |
|---|---|
| ![Features](screenshots/feature_importance.png) | ![Encyclopedia](screenshots/encyclopedia.png) |

> 📌 **TODO: Create a `screenshots/` folder in your repo and upload your app screenshots**

---

## 👩‍💻 Author

**Anitha Gowthami**
AI & ML Student — IIT Guwahati – IOT Academy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/TODO-your-linkedin-id)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Anitha-Gowthami-AIML)

> 📌 **TODO: Replace `TODO-your-linkedin-id` with your actual LinkedIn profile URL**

---

> ⭐ If you found this project helpful, please consider giving it a **star** on GitHub!
