import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np

app = Flask(__name__)

@app.route('/train', methods=['POST'])
def train():
    data = request.json['data']
    labels = request.json['labels']

    # Convert to NumPy arrays
    X = np.array(data)
    y = np.array(labels)

    # Define model
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(64, activation='relu', input_shape=(X.shape[1],)),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    model.fit(X, y, epochs=5, batch_size=32, verbose=1)

    # Serialize weights
    weights = [w.tolist() for w in model.get_weights()]
    return jsonify(weights)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
