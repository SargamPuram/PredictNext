from flask import Flask, request, jsonify
import pandas as pd
import joblib
from datetime import datetime

app = Flask(__name__)

# Load the trained model
rf_model = joblib.load('src/rf_model.pkl')

# Define current date for recency calculation (hardcoded or dynamically fetched)
current_date = pd.to_datetime('2024-12-07')  # Replace with today's date if needed

@app.route('/predict-next-purchase', methods=['POST'])
def predict_next_purchase():
    # Get data from request
    data = request.get_json()

    # Extract relevant data from the request
    customer_id = data['CustomerID']
    unit_price = data['unit_price']
    quantity = data['quantity']
    invoice_date = data['invoice_date']
    
    # Convert 'invoice_date' to datetime
    invoice_date = pd.to_datetime(invoice_date, format="%m/%d/%Y")

    # Calculate Recency (Days since the last purchase)
    recency = (current_date - invoice_date).days
    
    # Frequency: You can hardcode the frequency or get it from a database for real use cases
    frequency = 10  # Example: Assuming the customer made 10 purchases in the past

    # Revenue: Calculate revenue from the given unit_price and quantity
    revenue = unit_price * quantity

    # Create a DataFrame for prediction (mimicking the feature engineering from the training data)
    input_data = pd.DataFrame({
        'Recency': [recency],
        'Frequency': [frequency],
        'Revenue': [revenue]
    })

    # Predict the number of days until the next purchase
    predicted_days = rf_model.predict(input_data)

    # Calculate the predicted next purchase date
    predicted_next_purchase_date = invoice_date + pd.to_timedelta(predicted_days[0], unit='D')

    # Return the result as a JSON response
    return jsonify({
        "CustomerID": customer_id,
        "predicted_next_purchase_date": predicted_next_purchase_date.strftime('%Y-%m-%d')
    })

if __name__ == '__main__':
    app.run(debug=True)
