# Cyber-Threat-Detection-ML
<br>
# 🚀 Automated Cyber Threat Detection using Machine Learning

## 📌 Overview

End-to-end ML system to detect cyber intrusions from network traffic using the NSL-KDD dataset. Includes preprocessing, model training, evaluation, and a Flask API for inference.

## 🧠 Tech Stack

* Python, Pandas, NumPy
* Scikit-learn (Random Forest)
* Matplotlib, Seaborn
* Flask (API)

## 📊 Dataset

NSL-KDD (Kaggle) — download separately (not included in repo).

## ⚙️ Pipeline

1. Data Loading
2. Preprocessing (encoding categorical features, cleaning)
3. Train/Test split
4. Model Training (Random Forest)
5. Evaluation (Accuracy, Confusion Matrix)
6. Model Serialization (`model.pkl`)
7. API Deployment (Flask)

## 📈 Results

* Accuracy: ~99% on test split
* Strong detection of attack classes

## 🚀 How to Run

```bash
pip install -r requirements.txt
python step3_model.py
python app.py
```

Open: http://127.0.0.1:5000

## 📡 API

POST `/predict`
Body:

```json
{ "features": [ ... ] }
```

