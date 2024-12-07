import pandas as pd
from flask import Flask, request, jsonify
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the model
def load_model():
    model = joblib.load('src/xgb_model.pkl')
    return model

try:
    model = load_model()
    print("Model loaded successfully")
except FileNotFoundError:
    print("Model file not found")

# Preprocessing function
def preprocess_input(data):
    # Convert invoice_date to days since a reference date
    reference_date = datetime(2020, 1, 1)
    data['invoice_date'] = (datetime.strptime(data['invoice_date'], "%m/%d/%Y") - reference_date).days
    
    # Add derived features (make sure these match your training logic)
    data['RecencyCluster'] = data['days_since_reference'] // 10  # Example cluster calculation
    data['RevenueCluster'] = data['unit_price'] * data['quantity']  # Example revenue calculation
    data['OverallScore'] = data['RecencyCluster'] + data['RevenueCluster']  # Combined score
    
    # Ensure all expected features are included
    expected_features = [
        'CustomerID', 'Recency_x', 'Recency_y', 'Recency', 'RecencyCluster',
        'Frequency', 'FrequencyCluster', 'Revenue', 'RevenueCluster', 'OverallScore',
        'DayDiff_x', 'DayDiff2_x', 'DayDiff3_x', 'DayDiffMean_x_user', 'DayDiffStd_x_user',
        'DayDiff_y', 'DayDiff2_y', 'DayDiff3_y', 'DayDiffMean_y', 'DayDiffStd_y',
        'DayDiff_user', 'DayDiff2_user', 'DayDiff3_user', 'DayDiffMean', 'DayDiffStd',
        'DayDiff_order', 'DayDiff2_order', 'DayDiff3_order', 'DayDiffMean_x_order',
        'DayDiffStd_x_order', 'DayDiff', 'DayDiff2', 'DayDiff3', 'DayDiffMean_order',
        'DayDiffStd_order', 'Segment_High-Value', 'Segment_Low-Value', 'Segment_Mid-Value'
    ]

    # Convert input data to a DataFrame
    df = pd.DataFrame([data])

    # Add placeholder columns for any missing features
    for feature in expected_features:
        if feature not in df.columns:
            df[feature] = 0  # Default value for missing feature

    # Return the processed DataFrame with expected feature order
    return df[expected_features]

@app.route('/predict', methods=['POST'])
def predict_endpoint():
    try:
        # Get input JSON data
        input_data = request.get_json()

        # Preprocess the data
        processed_data = preprocess_input(input_data)

        # Predict using the loaded model
        prediction = model.predict(processed_data)

        return jsonify({'prediction': prediction.tolist()})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)
