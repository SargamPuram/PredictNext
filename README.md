# PredictNext
A machine-learning model to predict the next purchase date of a customer based on historical purchase data.

## Screenshots 
![Screenshot_20241207_234126](https://github.com/user-attachments/assets/a2a6cb18-a324-4e59-a434-8fe99b02eb58)

## Project Overview

This project involves analyzing the **Online Retail** dataset, which contains transactional data from an online retail store. The goal is to create a model that can predict the **days between the first and last purchase** for each customer. This helps in understanding customer loyalty and purchase patterns.

![image](https://github.com/user-attachments/assets/45d19df9-4cf3-4f27-8520-83f08c9e7ad6)


## Data Preprocessing

The dataset undergoes several preprocessing steps before any analysis:

1. **Loading the dataset**: The data is loaded from a CSV file and encoding is handled to ensure it is properly read.
2. **Handling Missing Data**: Rows with missing values in the 'InvoiceDate' and 'CustomerID' columns are removed.
3. **Label Encoding**: The `CustomerID` and `Country` columns are label encoded to convert categorical variables into numeric form.
4. **Feature Scaling**: Numerical features are standardized using `StandardScaler` to bring all features to the same scale.
5. **Date Conversion**: The `InvoiceDate` is converted to a `datetime` format for easier manipulation.

## Feature Engineering

Several features are engineered to understand customer behavior better:

- **Total Spend**: Calculated as the quantity of items multiplied by their unit price.
- **Recency**: The number of days since the customer’s last purchase.
- **Transaction Features**: Various aggregated features like total quantity, total spend, and purchase count are created.
- **First & Last Purchase Dates**: The first and last purchase dates for each customer are calculated to assess customer loyalty.

## Machine Learning

The goal is to predict the **Target** variable, which is the number of days between the customer's first and last purchase. The following steps are taken:

1. **Splitting Data**: The dataset is split into features (`X`) and target (`y`), and then further divided into training and testing sets.
2. **Model Training**: Although not explicitly mentioned in the code, you can train a machine learning model (e.g., regression model) on the training data.
3. **Prediction**: The model can then predict the **Target** variable for each customer.

## Results

After training the model, predictions can be used to identify customers who have been inactive for a long time, which could help the business in targeting retention campaigns.

## How to Run the Project

### Prerequisites

Before running the project, ensure you have the following installed:

- Python 3.x
- Required Python libraries (listed in `requirements.txt`)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/PredictNext.git
   cd PredictNext
2. **Create and activate a virtual environment:**  
   ```bash
    python -m venv venv  
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```
3. **Install dependencies:**  
   ```bash
   pip install -r requirements.txt  
   ```
4. Ensure the OnlineRetail.csv file is located in the ../data/ directory. If the file is not available, download it from here - https://www.kaggle.com/datasets/vijayuv/onlineretail.

5. Run the script:

```bash
python src/app.py
```

## Technologies Used
- Pandas: For data manipulation and analysis
- Matplotlib: For data visualization
- Seaborn: For advanced data visualization
- Scikit-learn: For machine learning and preprocessing
- Pickle: For saving and loading processed data

## Contributions 
Feel free to raise a PR for any improvements.




