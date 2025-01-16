import tensorflow as tf
from flask import Flask, jsonify
import pandas as pd
import numpy as np
import requests  # Importing requests library to send HTTP requests

app = Flask(__name__)

# Aggregator server URL
AGGREGATOR_URL = "http://127.0.0.1:5003/submit_weights"  # Change this to the correct URL if needed

@app.route('/train', methods=['POST'])
def train():
    # Load the preprocessed dataset
    file_path = r"C:\Users\makam\OneDrive\Desktop\FL\microservices\dataset\processed_dataset.csv"
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"Processed dataset not found at {file_path}"}), 404

    # Separate features and labels
    target_column = 'is_mal'
    if target_column in df.columns:
        y = df[target_column].values
        X = df.drop(columns=[target_column]).values
    else:
        return jsonify({"error": "Target column 'is_mal' missing in the dataset"}), 400

    # Define a local model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(128, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Train the model
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Serialize weights
    weights = model.get_weights()  # Get model weights as a list of NumPy arrays
    weights_list = [w.tolist() for w in weights]  # Convert each NumPy array to a list for JSON serialization

    # Send the weights to the aggregator
    try:
        response = requests.post(AGGREGATOR_URL, json=weights_list)
        if response.status_code == 200:
            return jsonify({"status": "Training completed and weights submitted to aggregator", "weights": weights_list})
        else:
            return jsonify({"status": "Training completed but failed to submit weights to aggregator", "error": response.text}), 500
    except requests.exceptions.RequestException as e:
        return jsonify({"status": "Training completed but failed to communicate with aggregator", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
