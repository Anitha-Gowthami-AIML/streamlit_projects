# 🫘 Bean Classification Model – 7 Varieties

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://bean-classifier-app-7-varieties.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![Status](https://img.shields.io/badge/Status-Live-brightgreen.svg)

> A multi-class Machine Learning classification web application that identifies **7 varieties of dry beans** from morphological features — built and deployed as part of the **AI & ML Program at IIT Guwahati – IOT Academy**.

---

## 🔗 Live Demo

👉 **[Try the App Here](https://bean-classifier-app-7-varieties.streamlit.app/)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Bean Varieties](#bean-varieties)
- [Dataset](#dataset)
- [Features Used](#features-used)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Models Trained](#models-trained)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Key Insights](#key-insights)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 📖 Overview

Identifying dry bean varieties manually is time-consuming and error-prone. This project uses **Machine Learning** to automatically classify dry beans into one of 7 registered varieties based on their physical and shape-based measurements.

The dataset originates from a real-world computer vision study at Selçuk University, Turkey, where **13,611 bean grain images** were captured with a high-resolution camera, and 16 morphological features were extracted per grain.

This end-to-end project covers data preprocessing, EDA, multi-model training, evaluation, and deployment as an interactive **Streamlit web application**.

---

## 🫘 Bean Varieties

The model classifies beans into one of these **7 registered varieties**:

| # | Variety | Description |
|---|---|---|
| 1 | **Seker** | Small, round-shaped bean |
| 2 | **Barbunya** | Medium-sized, spotted bean |
| 3 | **Bombay** | Large-sized bean |
| 4 | **Cali** | Medium-large, oval-shaped bean |
| 5 | **Dermason** | Small-sized, elongated bean |
| 6 | **Horoz** | Large, elongated hook-shaped bean |
| 7 | **Sira** | Medium-sized, oval/flat bean |

---

## 📂 Dataset

| Property | Details |
|---|---|
| **Source** | [UCI Machine Learning Repository – Dry Bean Dataset](https://archive.ics.uci.edu/dataset/602/dry+bean+dataset) |
| **Total Records** | 13,611 bean grain samples |
| **Total Features** | 16 morphological features |
| **Target Classes** | 7 (Seker, Barbunya, Bombay, Cali, Dermason, Horoz, Sira) |
| **License** | Creative Commons Attribution 4.0 (CC BY 4.0) |

> Citation: Koklu, M., & Ozkan, I. A. (2020). Multiclass classification of dry beans using computer vision and machine learning techniques. *Computers and Electronics in Agriculture*, 174, 105507.

---

## 🔬 Features Used

| # | Feature | Description |
|---|---|---|
| 1 | **Area** | Number of pixels within the bean boundary |
| 2 | **Perimeter** | Circumference length of the bean border |
| 3 | **Major Axis Length** | Length of the longest line through the bean |
| 4 | **Minor Axis Length** | Length of the shortest line through the bean |
| 5 | **Aspect Ratio** | Ratio of Major to Minor axis length |
| 6 | **Eccentricity** | Eccentricity of the equivalent ellipse |
| 7 | **Convex Area** | Pixels in the smallest convex polygon enclosing the bean |
| 8 | **Equivalent Diameter** | Diameter of circle with same area as the bean |
| 9 | **Extent** | Ratio of pixels in bounding box to bean area |
| 10 | **Solidity** | Ratio of pixels in convex shell to bean pixels |
| 11 | **Roundness** | Calculated using (4πA) / P² |
| 12 | **Compactness** | Measures roundness: Equivalent Diameter / Major Axis Length |
| 13 | **ShapeFactor1** | Shape descriptor 1 |
| 14 | **ShapeFactor2** | Shape descriptor 2 |
| 15 | **ShapeFactor3** | Shape descriptor 3 |
| 16 | **ShapeFactor4** | Shape descriptor 4 |

---

## 🛠️ Tech Stack

| Library | Purpose |
|---|---|
| **Python** | Core programming language |
| **Pandas** | Data loading, cleaning & manipulation |
| **NumPy** | Numerical operations |
| **Matplotlib & Seaborn** | EDA visualizations |
| **Scikit-learn** | Model training, evaluation & preprocessing |
| **Pickle / Joblib** | Saving & loading trained models |
| **Streamlit** | Interactive web app & deployment |

---

## 🤖 ML Pipeline

```
Raw Dataset (13,611 records × 16 features)
   │
   ▼
Data Cleaning & EDA
   │  ├── Null value check
   │  ├── Distribution plots
   │  └── Correlation analysis
   │
   ▼
Preprocessing
   │  ├── Label Encoding (target classes)
   │  └── Feature Scaling (StandardScaler)
   │
   ▼
Train-Test Split (80:20)
   │
   ▼
Model Training (5 Models)
   │  ├── Logistic Regression
   │  ├── Random Forest
   │  ├── SVM
   │  ├── KNN
   │  └── XGBoost
   │
   ▼
Model Evaluation
   │  └── Accuracy | Precision | Recall | F1-Score | Confusion Matrix
   │
   ▼
Best Model Saved with Pickle / Joblib
   │
   ▼
Streamlit App Deployment ✅
```

---

## 📊 Models Trained

| Model | Description |
|---|---|
| **Logistic Regression** | Baseline multi-class classifier |
| **Random Forest** | Ensemble of decision trees |
| **SVM** | Support Vector Machine with kernel trick |
| **KNN** | K-Nearest Neighbours classifier |
| **XGBoost** | Gradient boosted trees |

> 📌 **TODO: Add your best model's accuracy score here — e.g., "Best Model: XGBoost — Accuracy: 92.X%"**

---

## 📁 Project Structure

```
Bean_Classification_Model/
│
├── app.py                        # Main Streamlit application
├── bean_model.pkl                # Trained & saved ML model (TODO: confirm filename)
├── Dry_Bean_Dataset.xlsx         # Dataset (TODO: confirm your filename)
├── requirements.txt              # Python dependencies
└── README.md                     # Project documentation
```

> 📌 **TODO: Confirm your actual model file name and dataset file name**

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

## 💡 Key Insights

> 📌 **TODO: Fill in real findings from your EDA and model results — examples below to guide you:**

- 🏆 **[Best Model]** achieved the highest accuracy of **X%** among all 5 models
- 🫘 **Bombay** beans are the easiest to classify due to their distinctly large size
- 📏 **Major Axis Length** and **Area** were among the most important features for classification
- 🔁 **Dermason** and **Sira** varieties showed the most overlap, making them harder to distinguish
- 📊 The dataset had a reasonably balanced class distribution across all 7 varieties

---

## 📸 Screenshots

> 📌 **TODO: Add screenshots of your app**
>
> 1. Create a `screenshots/` folder in your repo
> 2. Take 3–4 screenshots from the live app
> 3. Upload and link them below:

```
![App Input](screenshots/input.png)
![Prediction Result](screenshots/result.png)
![Model Comparison](screenshots/model_comparison.png)
```

---

## 👩‍💻 Author

**Anitha Gowthami**
AI & ML Student — IIT Guwahati – IOT Academy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/TODO-your-linkedin-id)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Anitha-Gowthami-AIML)

> 📌 **TODO: Replace `TODO-your-linkedin-id` with your actual LinkedIn profile URL**

---

> ⭐ If you found this project helpful, please consider giving it a **star** on GitHub!
