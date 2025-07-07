from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import logging
from datetime import datetime
import os
import re
import nltk
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import sentiment_pipeline  
import webbrowser
import threading

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

# Initialize app
app = Flask(__name__)
CORS(app)

# Logging config
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Sentiment Analyzer class
class SentimentAnalyzer:
    def __init__(self):
        self.vader_analyzer = SentimentIntensityAnalyzer()
        self.stemmer = PorterStemmer()
        self.stop_words = set(stopwords.words('english'))
        self.custom_pipeline = None

        try:
            self.custom_pipeline = self.load_custom_pipeline()
            print("✅ Custom sentiment pipeline loaded successfully!")
        except Exception as e:
            print(f"❌ Failed to load custom pipeline: {e}")
            print("🔄 Falling back to VADER sentiment analysis")

    def load_custom_pipeline(self):
        if hasattr(sentiment_pipeline, 'predict_sentiment'):
            return sentiment_pipeline.predict_sentiment
        if hasattr(sentiment_pipeline, 'model') or hasattr(sentiment_pipeline, 'pipeline'):
            return sentiment_pipeline.model or sentiment_pipeline.pipeline
        if hasattr(sentiment_pipeline, 'SentimentModel'):
            return sentiment_pipeline.SentimentModel()
        return None

    def get_custom_sentiment(self, text):
        try:
            if callable(self.custom_pipeline):
                result = self.custom_pipeline(text)
                if isinstance(result, dict):
                    return {
                        'sentiment': result.get('sentiment', 'neutral'),
                        'confidence': result.get('confidence', 0.5)
                    }
                elif isinstance(result, tuple):
                    return {'sentiment': result[0], 'confidence': result[1]}
                else:
                    return {'sentiment': str(result), 'confidence': 0.8}
            elif hasattr(self.custom_pipeline, 'predict'):
                prediction = self.custom_pipeline.predict([text])[0]
                confidence = 0.8
                if hasattr(self.custom_pipeline, 'predict_proba'):
                    try:
                        proba = self.custom_pipeline.predict_proba([text])[0]
                        confidence = max(proba)
                    except:
                        pass
                return {'sentiment': prediction, 'confidence': confidence}
            elif hasattr(self.custom_pipeline, 'analyze'):
                return self.custom_pipeline.analyze(text)
        except Exception as e:
            print(f"❌ Custom pipeline error: {e}")
        return None

    def preprocess_text(self, text):
        text = text.lower()
        text = re.sub(r"http\S+|www\S+|https\S+", '', text)
        text = re.sub(r'@\w+|#\w+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return ' '.join(text.split())

    def get_vader_sentiment(self, text):
        scores = self.vader_analyzer.polarity_scores(text)
        compound_score = scores['compound']
        if compound_score >= 0.05:
            sentiment = 'positive'
            confidence = min(compound_score + 0.5, 1.0)
        elif compound_score <= -0.05:
            sentiment = 'negative'
            confidence = min(abs(compound_score) + 0.5, 1.0)
        else:
            sentiment = 'neutral'
            confidence = 0.5 + (0.5 - abs(compound_score))

        if sentiment in ['positive', 'negative'] and confidence < 0.75:
            sentiment = 'neutral'

        return {'sentiment': sentiment, 'confidence': confidence, 'raw_scores': scores}

    def analyze_sentiment(self, text):
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        if self.custom_pipeline:
            result = self.get_custom_sentiment(text)
            if result:
                print("✅ Using custom model sentiment analysis")
                return {
                    'sentiment': result['sentiment'],
                    'confidence': result['confidence'],
                    'analysis_method': 'Custom ML Pipeline',
                    'processed_text_length': len(text.strip()),
                    'original_text_length': len(text)
                }

        print("🔄 Using VADER sentiment analysis")
        clean = self.preprocess_text(text)
        result = self.get_vader_sentiment(text)
        return {
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'analysis_method': 'VADER (Fallback)',
            'raw_scores': result['raw_scores'],
            'processed_text_length': len(clean),
            'original_text_length': len(text)
        }

# Initialize sentiment analyzer
sentiment_analyzer = SentimentAnalyzer()

# === Routes ===

@app.route('/', methods=['GET'])
def home():
    if request.accept_mimetypes['application/json'] >= request.accept_mimetypes['text/html']:
        return jsonify({
            'status': 'running',
            'message': 'Sentiment API is live',
            'timestamp': datetime.now().isoformat()
        })
    try:
        return render_template("index.html")
    except:
        return jsonify({
            'status': 'running',
            'message': 'Sentiment API is live (HTML template missing)',
            'timestamp': datetime.now().isoformat()
        })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy'}), 200

@app.route('/analyze', methods=['POST'])
def analyze():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'No text provided'}), 400

        text = data['text'].strip()
        result = sentiment_analyzer.analyze_sentiment(text)
        logger.info(f"Analyzed: {result['sentiment']} (Confidence: {result['confidence']:.2f})")

        return jsonify({
            'sentiment': result['sentiment'],
            'confidence': result['confidence'],
            'analysis_method': result['analysis_method'],
            'raw_scores': result.get('raw_scores'),
            'text_length': result['original_text_length'],
            'timestamp': datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error during sentiment analysis: {e}")
        return jsonify({'error': 'Server error'}), 500

def open_browser():
    webbrowser.open_new('http://127.0.0.1:5000/')

# === Run the app ===
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    threading.Timer(1.5, open_browser).start()
    app.run(host='0.0.0.0', port=port, debug=True)
