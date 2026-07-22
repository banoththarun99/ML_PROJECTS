import pandas as pd

# loading the dataset
df = pd.read_csv("venv/Bank-Customer-Churn-Prediction/data/Customer_Churn_Records.csv")

#printing the first 5 rows of given dataset
print(df.head())

# finding the shape of the dataset
print(df.shape)

# finding the columns names 
print(df.columns)

# dataset information
df.info()

# finding the missing values
print(df.isnull().sum())

# finding the duplicate count
print(df.duplicated().sum())


