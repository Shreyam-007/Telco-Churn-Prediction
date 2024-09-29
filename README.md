# Customer Churn Prediction Using Telco Customer Churn Dataset

## Project Overview

This project aims to predict customer churn in a telecommunications company using the Telco Customer Churn dataset. By identifying customers who are likely to churn, the company can take proactive measures to retain them.

## Table of Contents

1. [Dataset](#dataset)
2. [Project Workflow](#project-workflow)
   - Data Exploration and Preprocessing
   - Exploratory Data Analysis (EDA)
   - Data Splitting
   - Model Development or Selection
   - Model Training with Cross Validation
   - Model Evaluation
   - Hyperparameter Tuning
   - Model Interpretation and Insights
3. [Results](#results)
4. [How to Use](#how-to-use)
5. [Dependencies](#dependencies)
6. [Conclusion](#conclusion)

## Dataset

The dataset used in this project is the Telco Customer Churn dataset, which contains customer details such as demographics, services subscribed, account information, and whether or not they churned.

### Dataset Features:
- CustomerID
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- PhoneService
- MultipleLines
- InternetService
- Contract, etc.

Target variable: Churn (whether a customer churned or not)

## Project Workflow
### 1. Data Preprocessing and Preprocessing

We first preprocess the data by:

- Handling missing values
- Encoding categorical variables
- Scaling numerical features
- Removing unnecessary columns (e.g., customerID)

### 2. Exploratory Data Analysis (EDA)

We performed EDA to understand the distribution of data, relationships between variables, and to visualize the imbalance in the target class (churn).

Key steps:

- Visualizing churn distribution
- Analyzing correlations between features
- Visualizing numerical and categorical features

### 3. Data Splitting
The dataset is split into training and testing sets to evaluate model performance.

### 4. Model Development or Selection
We built several machine learning models to predict customer churn:

- Logistic Regression
- Decision Tree
- Random Forest
- XGBoost
- LightGBM, etc

### 5. Model Training with Cross Validation
The models were trained on the training data, and we used techniques like cross-validation to ensure robust performance.

### 6. Model Evaluation
Since churn is imbalanced, we used metrics beyond accuracy, such as:

- Precision
- Recall
- F1-Score
- ROC-AUC
- Confusion Matrix

### 7. Hyperparameter Tuning
We used Grid Search and Random Search techniques to fine-tune hyperparameters and improve model performance.

### 8. Model Interpretation and Insights
We used feature importance and SHAP values to interpret the model.

- Feature Importance: Shows which features have the most impact on predictions.
- SHAP Values: Provides a deeper understanding of how individual predictions are made.

## Results
The XGBoost model showed the best performance in predicting churn with a high ROC-AUC score.
Top influential features: Contract type, tenure, monthly charges, and internet service were some of the key factors in predicting churn.

## How to Use
1. Clone this repository.
2. Install the required dependencies using the command:

>```pip install -r requirements.txt```

3. Run the Jupyter notebook or Python scripts to preprocess the dataset, train models, and evaluate them.

## Dependencies
Here are the main libraries used in this project:

- Pandas
- Numpy
- Scikit-learn
- Xgboost
- Shap
- Matplotlib
- Seaborn

Install them using:

>```pip install pandas numpy scikit-learn xgboost shap matplotlib seaborn```

## Conclusion
This project provides a comprehensive approach to predicting customer churn using machine learning. The model interpretation using SHAP values allows us to understand the key drivers of churn, helping the telecommunications company to retain customers effectively.