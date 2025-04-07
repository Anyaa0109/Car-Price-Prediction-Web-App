from flask import Flask, render_template, request, jsonify
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('linear_reg_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No JSON data received'}), 400

        year = int(data.get('year', 0))
        present_price = float(data.get('present_price', 0))
        kms_driven = int(data.get('kms_driven', 0))
        fuel_type = int(data.get('fuel_type', 0))
        seller_type = int(data.get('seller_type', 0))
        transmission = int(data.get('transmission', 0))
        owner = int(data.get('owner', 0))

        if not all([year, present_price, kms_driven]):
            return jsonify({'error': 'Missing required fields'}), 400

        input_features = np.array([[year, present_price, kms_driven, fuel_type, seller_type, transmission, owner]])
        predicted_price = model.predict(input_features)[0]

        return jsonify({'predicted_price': round(predicted_price, 2)})

    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
