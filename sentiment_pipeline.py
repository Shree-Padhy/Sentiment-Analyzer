# ===============================
# IMPORTS & SETUP
# ===============================
import pandas as pd
import numpy as np
import re
import string
import nltk
import joblib
import warnings
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from langdetect import detect
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Ignore warnings
warnings.filterwarnings("ignore")

# Download required NLTK resources
nltk.download('punkt')
nltk.download('stopwords')

# ===============================
# STEP 1: TEXT CLEANING & LANGUAGE DETECTION
# ===============================
def detect_language(text):
    try:
        return detect(text)
    except:
        return "unknown"

def clean_text(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    return text.lower()

try:
    reviews = pd.read_csv('reviews.csv', encoding='latin-1')
except Exception as e:
    print(f"Error loading the CSV file: {e}")
    exit()

reviews = reviews.drop(columns=['ID', 'Client_Name', 'Date'], errors='ignore')
reviews = reviews[reviews['Reviews'] != "No review found"]
reviews['language'] = reviews['Reviews'].apply(detect_language)
reviews['cleaned_reviews'] = reviews['Reviews'].apply(clean_text)

cleaned_dataset = reviews[['cleaned_reviews']].rename(columns={'cleaned_reviews': 'c_reviews'})
cleaned_dataset.to_csv('cleaned_reviews.csv', index=False)

print("✅ Cleaned dataset saved as 'cleaned_reviews.csv'.")

# ===============================
# STEP 2: VADER SENTIMENT LABELING
# ===============================
df = pd.read_csv('cleaned_reviews.csv')
analyzer = SentimentIntensityAnalyzer()

def preprocess_review(review):
    review = re.sub(r'[^a-zA-Z\s]', '', review)
    return review.lower()

def classify_review(review):
    score = analyzer.polarity_scores(review)
    if score['compound'] > 0.05:
        return 1  # Positive
    elif score['compound'] < -0.05:
        return -1  # Negative
    else:
        return 0  # Neutral

df['c_reviews'] = df['c_reviews'].apply(preprocess_review)
df['label'] = df['c_reviews'].apply(classify_review)
df[['c_reviews', 'label']].to_csv('labeled_reviews.csv', index=False)

print("✅ Labeled reviews saved as 'labeled_reviews.csv'.")

# ===============================
# STEP 3: MODEL TRAINING
# ===============================
df = pd.read_csv('labeled_reviews.csv')

def preprocess_text(text):
    text = re.sub(pattern='[^a-zA-Z]', repl=' ', string=text).lower()
    words = text.split()
    words = [word for word in words if word not in set(stopwords.words('english'))]
    ps = PorterStemmer()
    words = [ps.stem(word) for word in words]
    return ' '.join(words)

df['Processed_Text'] = df['c_reviews'].apply(preprocess_text)

cv = CountVectorizer(max_features=1500, ngram_range=(1, 2))
X = cv.fit_transform(df['Processed_Text']).toarray()
y = df['label'].values

joblib.dump(cv, 'cv.pkl')
print("✅ CountVectorizer saved as 'cv.pkl'.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=0)

classifiers = {
    'Naive Bayes': MultinomialNB(),
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(probability=True),
    'Logistic Regression': LogisticRegression()
}

metrics = {}

for name, model in classifiers.items():
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    metrics[name] = {
        'accuracy': accuracy_score(y_test, preds),
        'f1': f1_score(y_test, preds, average='weighted'),
        'precision': precision_score(y_test, preds, average='weighted'),
        'recall': recall_score(y_test, preds, average='weighted')
    }
    print(f"\n{name} - Accuracy: {metrics[name]['accuracy']:.4f}, F1: {metrics[name]['f1']:.4f}, "
          f"Precision: {metrics[name]['precision']:.4f}, Recall: {metrics[name]['recall']:.4f}")

best_model_name = max(metrics, key=lambda x: metrics[x]['accuracy'])
best_model = classifiers[best_model_name]
joblib.dump(best_model, 'model.pkl')

print(f"\n✅ Best model '{best_model_name}' saved as 'model.pkl'.")

# ===============================
# STEP 4: PREDICT FUNCTION FOR FLASK
# ===============================
model = joblib.load("model.pkl")
vectorizer = joblib.load("cv.pkl")

def predict_sentiment(text):
    processed = preprocess_text(text)
    X_new = vectorizer.transform([processed]).toarray()
    prediction = model.predict(X_new)[0]

    confidence = 0.8  # Default fallback
    if hasattr(model, 'predict_proba'):
        try:
            proba = model.predict_proba(X_new)[0]
            confidence = float(max(proba))

            # Get predicted class from model
            predicted_class = model.classes_[np.argmax(proba)]

            # Override positive/negative with neutral if confidence < 0.75
            if predicted_class in [1, -1] and confidence < 0.75:
                predicted_class = 0  # Neutral
                confidence = 1.0 - confidence  # Optional: adjust confidence

            prediction = predicted_class
        except Exception as e:
            print(f"Confidence override failed: {e}")
            pass

    sentiment_map = {1: 'positive', -1: 'negative', 0: 'neutral'}
    sentiment = sentiment_map.get(int(prediction), 'neutral')

    return {
        'sentiment': sentiment,
        'confidence': round(confidence, 3)
    }


# ===============================
# STEP 5: Optional Manual Test
# ===============================
if __name__ == "__main__":
    print("\n--- Real-Time Sentiment Prediction ---")
    while True:
        new_review = input("\nEnter a review (or type 'exit' to quit): ")
        if new_review.lower() == 'exit':
            print("Exiting sentiment prediction.")
            break
        result = predict_sentiment(new_review)
        print(f"Predicted Sentiment: {result['sentiment']} (Confidence: {result['confidence']})")
