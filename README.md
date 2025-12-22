# Anxiety Subtype Classification from Reddit Posts

This project focuses on automatically classifying anxiety-related social media posts into clinically meaningful subtypes using Natural Language Processing (NLP) and Machine Learning.

The anxiety subtypes considered are:
- Generalized Anxiety Disorder (GAD)
- Panic Disorder
- Social Anxiety

The system uses Reddit data, transformer-based text embeddings, sentiment analysis, keyword-based weak supervision, and multiple classifiers. An interactive Streamlit dashboard is provided for visualization and prediction.

---

##  Project Objectives

- Collect anxiety-related posts from Reddit using the Reddit API
- Apply weak supervision for initial subtype labeling
- Perform exploratory data analysis (EDA) to understand patterns
- Generate semantic embeddings using Sentence-BERT
- Train and compare multiple classification models
- Improve minority class performance (especially GAD)
- Build an interactive dashboard for visualization and inference

---

##  Type of Analytics Used

- **Descriptive Analytics**: Dataset statistics, sentiment analysis, keyword frequency
- **Diagnostic Analytics**: Confusion matrix, error analysis
- **Predictive Analytics**: Anxiety subtype classification

---

##  Project Structure

├── data/
│ ├── raw/ # Raw Reddit data
│ └── processed/ # Cleaned data and final predictions
│
├── features/
│ └── sbert_embeddings_all.npy
│
├── models/
│ ├── lr_model.pkl # Logistic Regression model
│ └── svm_model.pkl # SVM model (optional)
│
├── lets_do_it.py # Main data processing & modeling script
├── app.py # Streamlit dashboard
├── requirements.txt
├── README.md

##  Reddit API Setup (Important)

Create a Reddit app at:
https://www.reddit.com/prefs/apps

App settings:
- App type: **script**
- Redirect URI: `http://localhost:8080`

Store credentials securely using environment variables or Google Colab Secrets:
REDDIT_CLIENT_ID
REDDIT_SECRET
REDDIT_USER_AGENT

⚠️ Never hard-code credentials in code.

---

## 🚀 How to Run the Project

### 1️⃣ Install dependencies
```bash
pip install -r requirements.txt

### 2️⃣ Run data collection and modeling
python lets_do_it.py

### 3️⃣ Launch Streamlit dashboard
streamlit run app.py
