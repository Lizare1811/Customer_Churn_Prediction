#Churn predcition

#Model

#Libraries
import numpy as np
import pandas as pd
import missingno as ms
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

#Load dataset
df = pd.read_csv("C:/Users/meena/OneDrive/Desktop/Skola tech/ML/Customer Churn Project/Customer_Data.csv")
print(df.head())

#Check columns and its information
print(df.columns)
print(df.info())

#Check null values
print(df.isnull().sum())

#Visulaize missing values
ms.bar(df, figsize=(12,6),color="skyblue")
plt.title("Missing Values in Dataset")
plt.show()

#Fill missng values in categorical columns
categorical_cols = [
    'Value_Deal', 'Multiple_Lines', 'Internet_Type', 'Online_Security', 
    'Online_Backup', 'Device_Protection_Plan', 'Premium_Support', 
    'Streaming_TV', 'Streaming_Movies', 'Streaming_Music', 'Unlimited_Data'
]


#Fill missing values using mode without inplace
for col in categorical_cols:
    df[col] = df[col].fillna(df[col].mode()[0])

#Check after handling missing values
ms.bar(df, figsize=(12,6), color=["skyblue", "orange", "lightgreen", "violet"])
plt.title("Missing Values After Handling", fontsize=16)
plt.show()

#Drop rows where target columns are missing
df = df.dropna(subset=['Churn_Category', 'Churn_Reason'])

#Final check of missng values
print(df.isnull().sum())

#Final check of missing values with visualization
ms.bar(df, figsize=(14,6), color="lightgreen")
plt.title("Final Missing Values After Handling", fontsize=16)
plt.show()

#Identify columns to encode
#Detect categorical columns
cat_cols = df.select_dtypes(include='object').columns

#Separate binary and multi-category
binary_cols = [c for c in cat_cols if df[c].nunique() == 2]
multi_cols = [c for c in cat_cols if df[c].nunique() > 2]

print("Binary columns:", binary_cols)
print("Multi-category columns:", multi_cols)

#Binary categorical columns (Yes/No, 0/1 type)

# Gender
# Married
# Phone_Service
# Multiple_Lines
# Online_Security
# Online_Backup
# Device_Protection_Plan
# Premium_Support
# Streaming_TV
# Streaming_Movies
# Streaming_Music
# Unlimited_Data
# Paperless_Billing
# Customer_Status
# Churn_Category
#Churn_Reason     

# Multi-category columns (more than 2 categories)

# State
# Value_Deal
# Internet_Service
# Internet_Type
# Contract
# Payment_Method

#Label Encoding for binary columns
le = LabelEncoder()

#Encoding Each binary columns
binary_cols = [
    'Gender',
    'Married',
    'Phone_Service',
    'Multiple_Lines',
    'Online_Security',
    'Online_Backup',
    'Device_Protection_Plan',
    'Premium_Support',
    'Streaming_TV',
    'Streaming_Movies',
    'Streaming_Music',
    'Unlimited_Data',
    'Paperless_Billing',
    'Customer_Status'
]
#Common Label Encoding
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

#To Verify
print(df[binary_cols].head())

#one hot encoding for multi category columns
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

#Check columns(Confirms new dummy columns exist)
print(df.columns)

#Check shape(Columns count should increase)
print("Shape after encoding:", df.shape)

#View only onehot colums(Clearly shows dummy columns)
print(df.filter(like='Contract_').head())

#To check columns
print(df.columns[:15])

#To verify one hot columns exists
print(df.columns[15:30])

#In model.py, after all cleaning and encoding
df.to_csv("cleaned_churn_data.csv", index=False)
print("✅ Cleaned dataset saved successfully!")













