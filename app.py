from flask import Flask, render_template, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model using joblib
model = joblib.load("gradient_boosting_model.pkl")

# Define the selected features
FEATURES = ['solar_noon_dist', 'temperature', 'wind_dir', 'sky_cover', 
            'visibility', 'humidity', 'avg_wind_speed']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract required features from the form
        data = [float(request.form[feature]) for feature in FEATURES]

        # Predict
        prediction = model.predict(np.array(data).reshape(1, -1))

        return f"Predicted Power Generation: {prediction[0]:.2f} kW"
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
