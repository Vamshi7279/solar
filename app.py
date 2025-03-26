from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load trained model
with open("gradient_boosting_model.pkl", "rb") as file:
    model = pickle.load(file)

# Define the selected features
FEATURES = ['solar_noon_dist', 'temperature', 'wind_dir', 'sky_cover', 'visibility', 'humidity', 'avg_wind_speed']

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Extract only the required features
        data = [float(request.form[feature]) for feature in FEATURES]

        # Reshape and predict
        prediction = model.predict(np.array(data).reshape(1, -1))

        return f"Predicted Power Generation: {prediction[0]:.2f} kW"
    
    except Exception as e:
        return f"Error: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
