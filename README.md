# 🧠 Sentiment Analyzer

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-API-lightgrey)](https://flask.palletsprojects.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

A Flask-based API that performs sentiment analysis on user input using a custom machine learning model, with fallback to VADER. It includes modules for scraping reviews from Google Maps, preprocessing data, training classifiers, and serving real-time predictions via an API.

---

## 📌 Overview

This project combines **web scraping**, **natural language processing (NLP)**, and **machine learning** to create a sentiment classification system. It scrapes reviews, labels them using VADER, trains models, and serves predictions through a Flask API.

---

## 🖼️ Screenshot

![Screenshot 2025-07-07 113245](https://github.com/user-attachments/assets/4836465d-c316-4641-aafc-b20db80e2ada)

### Sentiment prediction interface rendered via Flask & HTML

---

## ✨ Features

- 🔎 Scrape reviews from Google Maps using Selenium  
- 🧹 Clean and label text with VADER sentiment analyzer  
- 🧠 Train and evaluate multiple classifiers  
- ✅ Automatically selects and uses the best model  
- ⚡ Real-time predictions via Flask API  
- 🔁 Fallback to VADER if the ML model fails  

---

## 🛠️ Tech Stack  
- **Frontend:** HTML5, CSS3  
- **Backend:** Python, Flask, Flask-CORS  
- **Sentiment Analysis:** VADER (NLTK), Scikit-learn  
- **Machine Learning:** Logistic Regression, Naive Bayes, SVM, Random Forest  
- **Web Scraping:** Selenium, WebDriver Manager  
- **Utilities:** CountVectorizer, Joblib, Logging  
- **Deployment:** Docker

---

## 📁 Project Structure

```plaintext
sentiment-analyzer/
├── requirements.txt            # 1 - Python dependencies (install first)
│
├── google_map_scraper.py       # 2 - Review scraper using Selenium
├── reviews.csv                 # 3 - Raw scraped review data (generated)
├── sentiment_pipeline.py       # 4.0 - ML pipeline: training, preprocessing, prediction
├── app.py                      # 4 - Main Flask app (calls sentiment_pipeline)
├── cleaned_reviews.csv         # 5 - Cleaned reviews after preprocessing
├── labeled_reviews.csv         # 6 - Reviews labeled using VADER
├── cv.pkl                      # 7 - CountVectorizer object (generated)
├── model.pkl                   # 8 - Trained ML model (generated)
│
├── templates/                  # HTML templates for web interface
│   └── index.html              # 9 - UI page
│
├── Deployment configuration.txt # Optional version pinning / deployment configs

```
---

## ⚙️ Setup Instructions

### 1. Clone the repository
git clone https://github.com/Shree-Padhy/Sentiment-Analyzer.git

cd Sentiment-Analyzer

### 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows
### OR
source venv/bin/activate  # On macOS/Linux

### 3. Install dependencies
pip install -r requirements.txt

---

## ▶️ Running the App

You can run the application in two ways:

### 🔹 Option 1: Use Provided Dataset (Quick Start)

```bash
python app.py
```

> This will load the pre-trained model (`model.pkl`) and vectorizer (`cv.pkl`) and start serving predictions instantly.

---

### 🔹 Option 2: Scrape & Train with Your Own Data

```bash
# 1. Scrape reviews from Google Maps
python google_map_scraper.py

# 2. Reviews will be saved in reviews.csv

# 3. Run the training pipeline (automatically triggered by app.py if model doesn't exist)
python app.py
```

> This flow trains a custom model based on fresh review data and starts the API server.

---

## 🧪 Model Training Workflow

- Scraped data is cleaned, normalized, and tokenized  
- VADER assigns sentiment labels for training  
- Text features extracted using `CountVectorizer`  
- Multiple models trained and evaluated  
- Best model saved as `model.pkl`, vectorizer as `cv.pkl`

---

## 🕵️ Web Scraper Module

- Script: `google_map_scraper.py`  
- Scrapes business reviews from Google Maps  
- Handles infinite scroll & dynamic content  
- Saves raw data to `reviews.csv`

---

## 🚀 Deployment

### Using Docker:

docker pull sriyapadhy/sentiment-analyzer:v1.0

docker run sriyapadhy/sentiment-analyzer:v1.0

---

## 👩‍💻 Author

Developed with ❤️ by **Sriya Padhy**  
🔗 [GitHub](https://github.com/Shree-Padhy)
🔗 [LinkedIn](https://www.linkedin.com/in/sriya-padhy-a21b7a260)


---
