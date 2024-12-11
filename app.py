from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load the saved model
model = joblib.load('linear_reg_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get form data
    fuel_type = int(request.form['Fuel_Type'])
    seller_type = int(request.form['Seller_Type'])
    transmission = int(request.form['Transmission'])
    present_price = float(request.form['Present_Price'])
    kms_driven = int(request.form['Kms_Driven'])
    owner = int(request.form['Owner'])
    age = int(request.form['Age'])

    # Prepare the input for prediction
    features = np.array([[present_price, kms_driven, owner, fuel_type, seller_type, transmission, age]])
    predicted_price = model.predict(features)[0]

    # Render template with prediction
    return render_template('index.html', prediction=round(predicted_price, 2))

if __name__ == '__main__':
    app.run(debug=True)
