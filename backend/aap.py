from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from flask_cors import CORS
import os

# NLTK setup
nltk.download('stopwords')
nltk.download('wordnet')
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Flask setup
app = Flask(__name__)
CORS(app)  # Allow frontend access

# Check if model files exist
if not os.path.exists('sentiment_model.h5') or not os.path.exists('tfidf_vectorizer.pkl'):
    raise FileNotFoundError("Model files not found")

try:
    # Load model and vectorizer
    model = load_model('sentiment_model.h5')
    with open('tfidf_vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
except Exception as e:
    raise Exception(f"Error loading model or vectorizer: {str(e)}")

# Cleaning function
pattern = re.compile(r'[^a-zA-Z]')
def clean_text(text):
    text = pattern.sub(' ', text)
    text = text.lower().split()
    text = [lemmatizer.lemmatize(word) for word in text if word not in stop_words]
    return ' '.join(text)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({"error": "Invalid input"}), 400

        text = data['text']
        cleaned = clean_text(text)
        vector = vectorizer.transform([cleaned]).toarray()
        vector_tensor = tf.convert_to_tensor(vector, dtype=tf.float32)
        prediction = model.predict(vector_tensor)
        label = "Positive" if prediction[0][0] >= 0.5 else "Negative"
        confidence = float(prediction[0][0]) if label == "Positive" else float(1 - prediction[0][0])
        return jsonify({
            'sentiment': label,
            'confidence': confidence
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5000)