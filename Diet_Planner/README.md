# NourishAI — Smart Recipe & Meal Plan Recommender

A beautiful, AI-powered meal planning app built with Streamlit.

## ML Techniques Used
| Model | Type | Purpose |
|---|---|---|
| K-Means Clustering | Unsupervised | Group users into dietary personas |
| SVD Matrix Factorization | Collaborative Filtering | Learn latent taste preferences |
| Cosine Similarity | Content-Based | Match recipe features to user profile |
| Weighted Hybrid Fusion | Ensemble | Combine all three signals |

## Project Structure
```
├── app.py                  # Streamlit app (main entry point)
├── recommender_engine.py   # ML logic & data generation
├── requirements.txt        # Python dependencies
└── README.md
```


## Features
- Personalised recipe recommendations based on diet, goals & allergens
- Hybrid ML engine (collaborative + content-based + clustering)
- 7-day meal plan with nutritional summaries
- YouTube video links for every recipe
- Beautiful dark food-themed UI with Playfair Display typography
- Calorie & prep-time filtering
- Cuisine and meal-type filtering
