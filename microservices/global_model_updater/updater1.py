import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import os
import logging
import requests

app = Flask(__name__)

# Define an absolute path for the model save location (Windows-compatible)
model_save_path = 'C:/Users/makam/OneDrive/Desktop/FL/microservices/global_model_updater/models/global_model.h5'

# Check if the directory exists, if not, create it
model_dir = os.path.dirname(model_save_path)
if not os.path.exists(model_dir):
    os.makedirs(model_dir)

# Set up logging
logging.basicConfig(level=logging.INFO)

# Update to handle 70 features in the model and prediction
NUM_FEATURES = 70  # Adjusted number of input features

@app.route('/update_model', methods=['GET'])
def update_model():
    """
    This endpoint gets the weights from the Trainer service, updates the model with the weights, and saves the model.
    """
    try:
        # GET weights from the Trainer service
        response = requests.get("http://localhost:5002/get_weights")

        # Check if the response is successful
        if response.status_code != 200:
            return jsonify({"status": "failed", "error": "Failed to get weights from trainer"}), 400

        weights = response.json().get('weights')

        if weights is None:
            return jsonify({"status": "failed", "error": "No weights found in the response"}), 400

        # Define the global model (adjust input shape based on number of features)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),  # Adjust input shape dynamically
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (is_mal)
        ])

        # Convert weights to NumPy arrays and set to the model
        weights = [np.array(weight) for weight in weights]

        # Ensure weights match the model layers
        if len(weights) == len(model.get_weights()):
            model.set_weights(weights)
        else:
            return jsonify({"status": "failed", "error": "Weight shape mismatch"}), 400

        # Save the model
        logging.info(f"Saving the model to {model_save_path}...")

        # Ensure directory exists and model is saved
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

        model.save(model_save_path)
        logging.info(f"Model saved successfully at {model_save_path}")

        return jsonify({"status": "Model updated and saved successfully!"})

    except Exception as e:
        logging.error(f"Error updating the model: {str(e)}")
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts input data for prediction and returns the prediction result using the updated global model.
    """
    data = request.json['data']  # Input data (numeric list)

    # Ensure that the input data is of the expected length (same as number of features)
    if len(data) != NUM_FEATURES:
        return jsonify({"status": "failed", "error": f"Expected {NUM_FEATURES} features, got {len(data)}"}), 400

    # Load the updated global model
    try:
        model = tf.keras.models.load_model(model_save_path)
    except Exception as e:
        return jsonify({"status": "failed", "error": f"Error loading model: {str(e)}"}), 500

    # Make prediction using the model
    try:
        input_data = np.array([data], dtype=np.float32)
        prediction = model.predict(input_data).tolist()
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

    return jsonify({"predictions": prediction})


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
