
# 🏥 Insurance Cost Predictor

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://app-insurance-cost-predictor.streamlit.app/)
![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML-orange.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

> A Machine Learning web application that predicts health insurance costs based on personal and lifestyle factors — built and deployed as part of the **AI & ML Program at IIT Guwahati – IOT Academy**.

---

## 🔗 Live Demo

👉 **[Try the App Here](https://app-insurance-cost-predictor.streamlit.app/)**

---

## 📌 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Dataset](#dataset)
- [Tech Stack](#tech-stack)
- [ML Pipeline](#ml-pipeline)
- [Project Structure](#project-structure)
- [How to Run Locally](#how-to-run-locally)
- [Key Insights](#key-insights)
- [Screenshots](#screenshots)
- [Author](#author)

---

## 📖 Overview

Insurance companies use several factors to determine premium costs for individuals. This project builds an **end-to-end predictive ML model** that estimates insurance charges based on demographic and health-related inputs.

The model is deployed as an interactive **Streamlit web app**, enabling users to instantly predict their insurance cost by simply filling in a form — no technical knowledge required.

---

## ✨ Features

- 🔮 **Real-time insurance cost prediction** based on user inputs
- 📊 **Interactive UI** built with Streamlit
- 🧠 **Trained ML model** using regression algorithms
- 📈 **Visual insights** into how each feature affects the predicted cost
- ☁️ **Deployed live** on Streamlit Cloud — accessible from any device

---

## 📂 Dataset

The project uses the popular **Medical Insurance Cost** dataset, which contains the following features:

| Feature | Description | Type |
|---|---|---|
| `age` | Age of the primary beneficiary | Numeric |
| `sex` | Gender of the individual | Categorical |
| `bmi` | Body Mass Index | Numeric |
| `children` | Number of dependents covered | Numeric |
| `smoker` | Smoking status (Yes/No) | Categorical |
| `region` | Residential region in the US | Categorical |
| `charges` | Annual medical insurance cost (**Target**) | Numeric |

> 📌 Source: [Kaggle - Medical Cost Personal Dataset](https://www.kaggle.com/datasets/mirichoi0218/insurance)

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| **Python 3.8+** | Core programming language |
| **Pandas & NumPy** | Data manipulation and analysis |
| **Matplotlib & Seaborn** | Exploratory data analysis & visualization |
| **Scikit-learn** | Model building, training, and evaluation |
| **Streamlit** | Web app frontend and deployment |
| **Pickle / Joblib** | Model serialization |

---

## 🤖 ML Pipeline

```
Raw Data
   │
   ▼
Data Cleaning & EDA
   │
   ▼
Feature Engineering
   │  ├── Label Encoding (sex, smoker)
   │  └── One-Hot Encoding (region)
   │
   ▼
Model Training
   │  ├── Linear Regression
   │  ├── Decision Tree Regressor
   │  ├── Random Forest Regressor
   │  └── Gradient Boosting Regressor
   │
   ▼
Model Evaluation (R², MAE, RMSE)
   │
   ▼
Best Model Selected & Saved
   │
   ▼
Streamlit App Deployment ✅
```

---

## 📁 Project Structure

```
Insurance_Cost_Predictor/
│
├── app.py                    # Main Streamlit application
├── model.pkl                 # Trained ML model (serialized)
├── insurance.csv             # Dataset
├── requirements.txt          # Python dependencies
├── Insurance_EDA.ipynb       # Exploratory Data Analysis notebook
└── README.md                 # Project documentation
```

---

## 🚀 How to Run Locally

### 1. Clone the Repository
```bash
git clone https://github.com/Anitha-Gowthami-AIML/streamlit_projects.git
cd streamlit_projects/Insurance_Cost_Predictor
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the Streamlit App
```bash
streamlit run app.py
```

### 4. Open in Browser
```
http://localhost:8501
```

---

## 💡 Key Insights

From the data analysis and model training, the following patterns emerged:

- 🚬 **Smoking is the #1 cost driver** — smokers pay **3–4x more** than non-smokers with identical profiles
- 📅 **Age has a strong positive correlation** with insurance charges — costs rise steadily with age
- ⚖️ **High BMI (obesity)** significantly increases predicted costs, especially when combined with smoking
- 👶 **Number of children** has a relatively minor effect compared to smoking or BMI
- 🗺️ **Region** has a small but measurable impact on charges

---

## 📸 Screenshots

> *(Add screenshots of your app here)*
>
> **Tip:** Replace the placeholders below with actual screenshots from your live app.

| Input Form | Prediction Result |
|---|---|
| ![Input](screenshots/input.png) | ![Result](screenshots/result.png) |

---

## 👩‍💻 Author

**Anitha Gowthami**
AI & ML Student — IIT Guwahati – IOT Academy

[![LinkedIn](https://img.shields.io/badge/LinkedIn-Connect-blue?logo=linkedin)](https://www.linkedin.com/in/your-linkedin-id)
[![GitHub](https://img.shields.io/badge/GitHub-Follow-black?logo=github)](https://github.com/Anitha-Gowthami-AIML)

---

## 📜 License

This project is licensed under the **MIT License** — feel free to use, share, and build upon it with attribution.

---

> ⭐ If you found this project helpful, please consider giving it a **star** on GitHub!
