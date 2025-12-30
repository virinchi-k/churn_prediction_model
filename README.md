# Customer Churn Prediction Model

## Overview

This project implements ML solutions to predict customer churn for a bank using various classification algorithms. Models analyze customer behavioral data to identify patterns that indicate whether a customer is likely to churn (leave the bank).

## Dataset

The dataset used is `Churn_Modelling.csv`. This contains information about 10,000 customers with the following features:

- **RowNumber**: Row index
- **CustomerId**: Unique customer identifier
- **Surname**: Customer's surname
- **CreditScore**: Customer's credit score
- **Geography**: Customer's country (France, Spain, Germany)
- **Gender**: Customer's gender (Male/Female)
- **Age**: Customer's age
- **Tenure**: Number of years the customer has been with the bank
- **Balance**: Account balance
- **NumOfProducts**: Number of bank products the customer uses
- **HasCrCard**: Whether the customer has a credit card (1 = Yes, 0 = No)
- **IsActiveMember**: Whether the customer is an active member (1 = Yes, 0 = No)
- **EstimatedSalary**: Customer's estimated salary
- **Exited**: Target variable indicating churn (1 = Churned, 0 = Active)

## Project Structure

churn_prediction_model

├── Churn_Modelling.csv              # Dataset file
├── churn_prediction_model.ipynb     # Main Jupyter notebook with complete analysis
├── README.md                         # Project documentation
├── requirements.txt                  # Python dependencies


## Installation and Requirements

### Prerequisites

- Python 3.7+
- Jupyter Notebook
- Required Python packages (install via pip):

### Dependencies

- **pandas**: Data manipulation and analysis
- **numpy**: Numerical computing
- **matplotlib**: Data visualization
- **seaborn**: Statistical data visualization
- **scikit-learn**: ML algorithms and evaluation metrics
- **ydata-profiling**: Automated exploratory data analysis

## Methodology

### 1. Data Preprocessing

- **Data Cleaning**: Checked for missing values and data types
- **Feature Encoding**:
  - Label encoding for Gender (Male=1, Female=0)
  - One-hot encoding for Geography (creating dummy variables for Germany and Spain)
- **Feature Selection**: Selected relevant features excluding identifiers and non-predictive columns
- **Feature Scaling**: Applied StandardScaler for numerical features

### 2. Exploratory Data Analysis (EDA)

- Analyzed data distributions, correlations, and missing values
- Visualized feature importance using Random Forest

### 3. Model Development

Implemented and compared multiple classification algorithms:

#### Random Forest Classifier
- **Parameters**: n_estimators=100, random_state=42
- **Accuracy**: ~86%
- **Best performing model in initial evaluation**

#### Logistic Regression
- **Parameters**: random_state=42
- **Accuracy**: ~81%
- **Baseline model for comparison**

#### Support Vector Machine (SVM)
- **Parameters**: kernel='linear', random_state=42
- **Accuracy**: ~80%
- **Issues with class imbalance affecting precision/recall**

#### K-Nearest Neighbors (KNN)
- **Parameters**: n_neighbors=5
- **Accuracy**: ~83%
- **Good performance on balanced metrics**

#### Gradient Boosting Classifier
- **Parameters**: n_estimators=100, random_state=42
- **Accuracy**: ~86%
- **Competitive performance with Random Forest**

### 4. Feature Engineering

Attempted advanced feature engineering techniques:

- **BalanceZero**: Binary feature indicating zero balance accounts
- **AgeGroup**: Categorical age buckets (18-25, 26-35, etc.)
- **BalanceToSalaryRatio**: Ratio of balance to estimated salary
- **ProductUsage**: Interaction between NumOfProducts and IsActiveMember
- **TenureGroup**: Categorical tenure buckets
- **Male_Germany/Male_Spain**: Interaction features between gender and geography

### 5. Model Evaluation

Evaluated models using:
- **Accuracy Score**: Overall prediction accuracy
- **Confusion Matrix**: True positives, false positives, etc.
- **Classification Report**: Precision, recall, F1-score for each class
- **Feature Importance**: Identified key predictors using Random Forest

## Results

### Model Performance Comparison

| Model | Accuracy | Key Observations |
|-------|----------|------------------|
| Random Forest | 86% | Best initial performance, good balance of precision/recall |
| Gradient Boosting | 86% | Similar to Random Forest, slightly better on some metrics |
| KNN | 83% | Solid performance, good for interpretation |
| Logistic Regression | 81% | Baseline model, interpretable coefficients |
| SVM | 80% | Affected by class imbalance, lower precision/recall |

### Key Insights

1. **Important Features** (in order of importance):
   - Age
   - Balance
   - NumOfProducts
   - EstimatedSalary
   - CreditScore
   - IsActiveMember

2. **Class Imbalance**: The dataset shows imbalance between churned and active customers, affecting model performance on minority class predictions.

3. **Feature Engineering**: Additional engineered features did not significantly improve model performance, suggesting the original features were sufficiently informative.


### Making Predictions

To use the trained model for new predictions:

```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load and preprocess new data following the same steps
# Use the trained model to make predictions
predictions = model.predict(new_data_scaled)
```


## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.