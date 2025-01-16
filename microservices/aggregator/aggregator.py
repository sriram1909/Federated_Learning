from flask import Flask, request, jsonify
import numpy as np
import requests

app = Flask(__name__)

# Store received weights from clients (local trainers)
client_weights = []

@app.route('/submit_weights', methods=['POST'])
def submit_weights():
    """
    Endpoint for receiving weights from clients (local trainers).
    """
    global client_weights
    try:
        # Parse weights from the request
        weights = request.get_json()

        if not weights:
            return jsonify({"status": "failed", "error": "No weights received"}), 400

        # Convert each layer's weights to a NumPy array for easier aggregation
        client_weights.append([np.array(layer) for layer in weights])
        return jsonify({"status": "success", "message": "Weights received successfully"})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 400


@app.route('/aggregate', methods=['GET'])
def aggregate_weights():
    """
    Aggregates all received weights using averaging and returns the global weights.
    """
    global client_weights

    if not client_weights:
        return jsonify({"status": "failed", "error": "No weights to aggregate"}), 400

    try:
        # Perform layer-wise aggregation (averaging)
        aggregated_weights = [
            np.mean([client[layer_idx] for client in client_weights], axis=0)
            for layer_idx in range(len(client_weights[0]))
        ]

        # Convert the aggregated weights to a list for JSON serialization (to solve the ndarray serialization issue)
        aggregated_weights_list = [layer.tolist() for layer in aggregated_weights]

        # Send aggregated weights to the updater service
        response = send_weights_to_updater(aggregated_weights_list)

        # Reset the client weights for the next round
        client_weights = []

        return jsonify({"status": "success", "global_weights": aggregated_weights_list, "updater_response": response.json()})
    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route('/reset_weights', methods=['GET'])
def reset_weights():
    """
    Reset the stored client weights for a fresh round of federated learning.
    """
    global client_weights
    client_weights = []
    return jsonify({"status": "success", "message": "All weights have been reset"})


def send_weights_to_updater(weights):
    """
    Sends the aggregated weights to the 'update_model' endpoint of the updater service.
    """
    url = "http://localhost:5004/update_model"  # The updater service URL
    payload = {"weights": weights}

    headers = {"Content-Type": "application/json"}

    # Send POST request to the updater service
    response = requests.post(url, json=payload, headers=headers)

    return response


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5003, threaded=True)

