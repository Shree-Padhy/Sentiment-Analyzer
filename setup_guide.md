# Sentiment Analysis Web Application

A complete full-stack sentiment analysis application with a modern web interface and Flask API backend.

## 🚀 Quick Start

### Local Development

1. **Clone/Download the files**
   ```bash
   # Create project directory
   mkdir sentiment-analyzer
   cd sentiment-analyzer
   
   # Save the Flask backend as app.py
   # Save the HTML frontend as index.html
   # Save requirements.txt
   ```

2. **Set up Python environment**
   ```bash
   # Create virtual environment
   python -m venv venv
   
   # Activate virtual environment
   # On Windows:
   venv\Scripts\activate
   # On Mac/Linux:
   source venv/bin/activate
   
   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Run the backend**
   ```bash
   python app.py
   ```
   The API will be available at `http://localhost:5000`

4. **Open the frontend**
   - Open `index.html` in your web browser
   - Or serve it with a simple HTTP server:
   ```bash
   # Python 3
   python -m http.server 8000
   # Then visit http://localhost:8000
   ```

## 🔧 Integration with Your ML Pipeline

The current setup uses VADER sentiment analysis. To integrate your existing ML pipeline:

### Option 1: Replace the analyzer
```python
# In app.py, modify the SentimentAnalyzer class
def load_trained_model(self):
    import joblib
    return joblib.load('your_model.pkl')

def get_ml_sentiment(self, text):
    prediction = self.ml_model.predict([text])[0]
    