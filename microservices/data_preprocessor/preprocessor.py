from flask import Flask, request, jsonify
import pandas as pd
import os

app = Flask(__name__)

@app.route('/preprocess', methods=['POST'])
def preprocess():
    data = request.json['data']
    df = pd.DataFrame(data)

    # Preprocessing steps
    df.fillna(0, inplace=True)
    numerical_columns = ['flow_duration', 'Rate', 'IAT', 'Tot size']
    df[numerical_columns] = (df[numerical_columns] - df[numerical_columns].mean()) / df[numerical_columns].std()

    processed_data = df.to_dict(orient='records')
    return jsonify(processed_data)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
