import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import pandas as pd
import os

app = Flask(__name__)

# Directory where the model will be saved
model_save_path = '/models/trained_model.h5'

# Define the number of features
NUM_FEATURES = 70  # Update to 70 features as per your dataset

# Load the dataset for training
def load_dataset():
    """
    Loads the dataset and returns the features and labels.
    Assumes the dataset is a CSV file with the last column being the label 'is_mal'.
    """
    try:
        # Load the dataset
        file_path = r"C:\Users\makam\OneDrive\Desktop\FL\microservices\dataset\processed_dataset.csv"
        df = pd.read_csv(file_path)  # Ensure this file exists and contains the correct data
        
        # Features are all columns except the last one (is_mal)
        X_train = df.iloc[:, :-1].values  # All columns except the last column
        y_train = df.iloc[:, -1].values  # Last column (is_mal)
        
        return X_train, y_train
    except Exception as e:
        raise ValueError(f"Error loading dataset: {str(e)}")

@app.route('/train', methods=['POST'])
def train_model():
    """
    This endpoint trains the model using the dataset and returns the weights.
    """
    try:
        # Load the dataset
        X_train, y_train = load_dataset()

        # Define the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (is_mal)
        ])

        # Compile the model
        model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

        # Train the model
        model.fit(X_train, y_train, epochs=5, batch_size=32)

        # Get the weights of the model
        weights = model.get_weights()

        # Save the trained model
        model.save(model_save_path)

        # Convert the weights into a list of lists and send back in the response
        weights_list = [weight.tolist() for weight in weights]

        return jsonify({"status": "Training completed successfully", "weights": weights_list})

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/get_weights', methods=['GET'])
def get_weights():
    """
    This endpoint returns the weights of the trained model.
    """
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_save_path)

        # Get the weights of the model
        weights = model.get_weights()

        # Convert the weights into a list of lists
        weights_list = [weight.tolist() for weight in weights]

        return jsonify({"status": "success", "weights": weights_list})

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    """
    This endpoint accepts input data and returns the prediction.
    """
    try:
        # Load the trained model
        model = tf.keras.models.load_model(model_save_path)

        # Get the data from the request
        data = request.get_json()

        # Ensure the input has 70 features
        if len(data) != NUM_FEATURES:
            return jsonify({"status": "failed", "error": f"Expected {NUM_FEATURES} features, got {len(data)}"}), 400

        # Convert the input data into a numpy array
        input_data = np.array(data).reshape(1, -1)  # Reshape to match model input

        # Make the prediction
        prediction = model.predict(input_data)[0][0]

        # Return the result
        return jsonify({"status": "success", "prediction": prediction})

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
