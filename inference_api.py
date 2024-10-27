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
