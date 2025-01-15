from flask import Flask, request, jsonify
import tensorflow as tf
import numpy as np
import os

app = Flask(__name__)

@app.route('/update_model', methods=['POST'])
def update_model():
    global_weights = request.get_json()
    try:
        # Define the model architecture
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
            tf.keras.layers.Dense(32, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])
        
        # Convert JSON weights to NumPy arrays
        numpy_weights = [np.array(w) for w in global_weights]
        
        # Ensure the weights match the model's structure
        model.set_weights(numpy_weights)
        
        # Save the updated model
        os.makedirs('models', exist_ok=True)
        model.save('models/global_model.h5')
        
        return jsonify({"status": "Model updated and saved!"})
    except ValueError as ve:
        return jsonify({"status": "Failed", "error": f"ValueError: {str(ve)}"}), 400
    except Exception as e:
        return jsonify({"status": "Failed", "error": f"Exception: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
