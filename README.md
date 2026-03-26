Customer Churn Category Prediction using Machine Learning

Project Overview

This project predicts the category/reason for customer churn using machine learning techniques. Instead of only identifying whether a customer will churn, it classifies the underlying reason for churn, helping businesses take targeted actions.

Tech Stack

1)Python

2)Pandas

3) NumPy

4)Scikit-learn

5)Matplotlib / Seaborn



Dataset

The dataset contains customer-related information such as demographics, account details, and service usage.

Key Features:

1)Gender

2)Tenure

3)Monthly Charges

4)Contract Type

5)Payment Method

6)Internet Service

Target Variable:

Churn_Category (e.g., Competitor, Price, Service Issues, etc.)


Methodology

1)Data cleaning and preprocessing

2)Handling missing values

3)Encoding categorical variables

4)Feature selection

5)Splitting dataset into training and testing sets

6)Model training and evaluation



Models Used

1)Logistic Regression

2)Decision Tree

3)Random Forest



Results

 Logistic Regression Accuracy: 61.38% 
 
 Decision Tree Accuracy: 48.12%
 
 Random Forest Accuracy: 61.67%
 
Analysis

The model shows moderate performance, indicating that customer churn prediction is a complex problem influenced by multiple factors such as customer behavior and service usage.



How to Run

1. Clone the repository:
   git clone https://github.com/your-username/customer-churn-prediction.git

2. Navigate to the project folder:
   cd customer-churn-prediction

3. Install required libraries:
   pip install -r requirements.txt

4. Run the project:
   python main.py

Output

The model predicts the category of customer churn based on input features.

Example Output:

Predicted Churn Category: Competitor


Future Improvements

1)Improve accuracy using feature engineering

2)Handle class imbalance (SMOTE)

3)Hyperparameter tuning

4)Use advanced models like XGBoost or LightGBM

5)Deploy as a web application


Conclusion

This project demonstrates how machine learning can be applied to solve real-world business problems like customer retention. It also highlights the challenges involved in predicting customer behavior.

