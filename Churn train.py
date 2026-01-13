#Train File

#Import Libraries
import pandas as pd
import numpy as np

#Model training & evaluation
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

#Load the cleaned dataset
df = pd.read_csv("cleaned_churn_data.csv")
print("Dataset loaded successfully!")

#To check
print(df.shape)
print(df.head())

#Create a single target column from one-hot encoded churn columns
#1.Identify all one-hot encoded churn columns
churn_cols = [col for col in df.columns if col.startswith('Churn_Category_')]
#2.Create a single target column by taking the column with the max value (1) for each row
df['Churn_Category_Target'] = df[churn_cols].idxmax(axis=1)
#3.Remove the prefix to get clean category names
df['Churn_Category_Target'] = df['Churn_Category_Target'].str.replace('Churn_Category_', '')

#After creating the target column
print(df[['Churn_Category_Target']].head())

#For confirmation
print("Target column 'Churn_Category_Target' created successfully!")
print(df['Churn_Category_Target'].value_counts())

# Selected features
selected_features = [
    'Age', 
    'Tenure_in_Months', 
    'Number_of_Referrals', 
    'Monthly_Charge', 
    'Total_Revenue', 
    'Customer_Status'
]

# Target column(single churn category)
target = 'Churn_Category_Target'

#Define X and y
X = df[selected_features]
y = df[target]

#Verify
print("Features (X) shape:", X.shape)
print("Target (y) shape:", y.shape)
print("\nFeature sample:")
print(X.head())
print("\nTarget sample:")
print(y.head())

#Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

#Verify the shapes
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)


#To scale numerical features for Logistic Regression
from sklearn.preprocessing import StandardScaler
num_features = ['Age', 'Tenure_in_Months', 'Number_of_Referrals', 'Monthly_Charge', 'Total_Revenue']

scaler = StandardScaler()
X_train[num_features] = scaler.fit_transform(X_train[num_features])
X_test[num_features] = scaler.transform(X_test[num_features])

print("Numerical features scaled successfully!")

#Logistic Regression
logreg = LogisticRegression(max_iter=2000)  # Create Logistic Regression model, allow up to 2000 iterations for convergence
logreg.fit(X_train, y_train)                # Train the model on training data (features X_train, target y_train)
y_pred_lr = logreg.predict(X_test)          # Predict the target for test data (X_test)
print("=== Logistic Regression ===")        # Print header to show results belong to Logistic Regression
print("Accuracy:", accuracy_score(y_test, y_pred_lr))  # Calculate and print accuracy of model on test data

#Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("=== Decision Tree ===")
print("Accuracy:", accuracy_score(y_test, y_pred_dt))

#Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("=== Random Forest ===")
print("Accuracy:", accuracy_score(y_test, y_pred_rf))

#New prediction

new_customer = pd.DataFrame({
    'Age': [45],
    'Tenure_in_Months': [24],
    'Number_of_Referrals': [2],
    'Monthly_Charge': [70],
    'Total_Revenue': [1800],
    'Customer_Status': [0]
})

#scaled numerical features
new_customer[num_features] = scaler.transform(new_customer[num_features])

#Predict churn category
prediction = rf.predict(new_customer)
print("Predicted Churn Category:", prediction[0])



