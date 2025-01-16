import tensorflow as tf
from flask import Flask, request, jsonify
import numpy as np
import requests

app = Flask(__name__)

# Define the number of features based on your dataset
NUM_FEATURES = 45  # Adjust this if necessary


@app.route('/update_model', methods=['POST'])
def update_model():
    """
    This endpoint gets the weights from the Aggregator, updates the model with the weights, and saves the model.
    """
    try:
        # GET aggregated weights from the Aggregator service
        response = requests.get("http://localhost:5003/aggregate")

        # Check if the response is successful
        if response.status_code != 200:
            return jsonify({"status": "failed", "error": "Failed to get aggregated weights from aggregator"}), 400

        # Get the aggregated weights from the response
        aggregated_weights = response.json().get('global_weights')

        if not aggregated_weights:
            return jsonify({"status": "failed", "error": "No aggregated weights found in response"}), 400

        # Define the global model (adjust input shape based on number of features)
        model = tf.keras.Sequential([
            tf.keras.layers.Dense(128, activation='relu', input_shape=(NUM_FEATURES,)),  # Adjust input shape
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')  # Output layer for binary classification (is_mal)
        ])

        # Convert aggregated weights to NumPy arrays and set to the model
        aggregated_weights = [np.array(weight) for weight in aggregated_weights]
        model.set_weights(aggregated_weights)

        # Save the updated global model to the disk
        model.save('/models/global_model.h5')

        return jsonify({"status": "Model updated and saved successfully!"})

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    """
    Accepts input data for prediction and returns the prediction result using the updated global model.
    """
    data = request.json['data']  # Input data for prediction

    if not data:
        return jsonify({"status": "failed", "error": "Input data missing"}), 400

    # Prepare the input data: Extract numerical values from input (features)
    input_data = [
        data['flow_duration'],
        data['Header_Length'],
        data['Source Port'],
        data['Destination Port'],
        data['Protocol Type'],
        data['Duration'],
        data['Rate'],
        data['Srate'],
        data['Drate'],
        data['fin_flag_number'],
        data['syn_flag_number'],
        data['rst_flag_number'],
        data['psh_flag_number'],
        data['ack_flag_number'],
        data['urg_flag_number'],
        data['ece_flag_number'],
        data['cwr_flag_number'],
        data['ack_count'],
        data['syn_count'],
        data['fin_count'],
        data['urg_count'],
        data['rst_count'],
        data['max_duration'],
        data['min_duration'],
        data['sum_duration'],
        data['average_duration'],
        data['std_duration'],
        data['CoAP'],
        data['HTTP'],
        data['HTTPS'],
        data['DNS'],
        data['Telnet'],
        data['SMTP'],
        data['SSH'],
        data['IRC'],
        data['TCP'],
        data['UDP'],
        data['DHCP'],
        data['ARP'],
        data['ICMP'],
        data['IGMP'],
        data['IPv'],
        data['LLC'],
        data['Tot sum'],
        data['Min'],
        data['Max'],
        data['AVG'],
        data['Std'],
        data['Tot size'],
        data['IAT'],
        data['Number'],
        data['MAC'],
        data['Magnitue'],
        data['Radius'],
        data['Covariance'],
        data['Variance'],
        data['Weight'],
        data['DS status'],
        data['Fragments'],
        data['Sequence number'],
        data['Protocol Version'],
        data['flow_idle_time'],
        data['flow_active_time'],
        data['label'],
        data['subLabel'],
        data['subLabelCat']
    ]

    try:
        # Load the updated global model
        model = tf.keras.models.load_model('/models/global_model.h5')

        # Make a prediction
        prediction = model.predict(np.array([input_data])).tolist()

        return jsonify({"predictions": prediction})

    except Exception as e:
        return jsonify({"status": "failed", "error": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5004)
