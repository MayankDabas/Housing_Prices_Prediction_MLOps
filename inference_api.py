"""
This script sets up a simple Flask-based API for serving predictions from 
a trained RandomForest model. The model is loaded from a serialized file, 
and predictions are made based on input features provided via a POST request.

Endpoints:
1. /predict (POST): Accepts JSON input with feature values and returns the model's prediction.

Expected Input Format:
{
    "features": [feature1, feature2, ..., featureN]
}

Example Output Format:
{
    "prediction": predicted_value
}
"""

from flask import Flask, request, jsonify
import joblib

with open('final_model.pkl', 'rb') as f:
    model = joblib.load(f)

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    if not data:
        return jsonify({'error': 'No input data provided'}), 400

    try:
        features = data['features']
        prediction = model.predict([features])
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
