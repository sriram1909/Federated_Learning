import tensorflow as tf
import numpy as np
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/update_model', methods=['POST'])
def update_model():
    global_weights = request.json['weights']
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.set_weights([np.array(w) for w in global_weights])
    model.save('/models/global_model.h5')
    return jsonify({"status": "Model updated and saved!"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
