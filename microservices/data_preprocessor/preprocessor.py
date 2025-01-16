from flask import Flask, request, jsonify
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os

app = Flask(__name__)

@app.route('/preprocess', methods=['GET'])
def preprocess():
    # Load the dataset
    file_path = r"C:\Users\makam\OneDrive\Desktop\FL\microservices\dataset\dataset.csv"

    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return jsonify({"error": f"Dataset not found at {file_path}"}), 404

    # Separate the target column
    target_column = 'is_mal'
    if target_column in df.columns:
        y = df[target_column]
        df.drop(columns=[target_column], inplace=True)
    else:
        y = None

    # Identify numerical and categorical columns
    numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
    categorical_columns = df.select_dtypes(include=['object']).columns

    # Normalize numerical columns
    scaler = StandardScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Encode categorical columns
    encoders = {}
    for col in categorical_columns:
        encoder = LabelEncoder()
        df[col] = encoder.fit_transform(df[col])
        encoders[col] = encoder

    # Add the target column back, if available
    if y is not None:
        df[target_column] = y

    # Ensure the directory exists
    output_directory = r"C:\Users\makam\OneDrive\Desktop\FL\microservices\dataset"
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Save processed data for other services
    output_path = os.path.join(output_directory, "processed_dataset.csv")
    df.to_csv(output_path, index=False)

    return jsonify({"status": "Preprocessing completed", "processed_file": output_path})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001)
