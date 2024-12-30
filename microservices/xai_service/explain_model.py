from flask import Flask, jsonify, request
import shap
import pandas as pd
from tensorflow.keras.models import load_model

app = Flask(__name__)

@app.route('/explain', methods=['POST'])
def explain():
    data = pd.DataFrame(request.json['data'])
    model = load_model('/models/global_model.h5')

    explainer = shap.KernelExplainer(model.predict, data)
    shap_values = explainer.shap_values(data)

    return jsonify(shap_values=shap_values)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5005)
