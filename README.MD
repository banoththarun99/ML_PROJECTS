# Employee Salary Prediction Using Machine Learning

## Overview

Employee compensation depends on multiple factors such as professional experience, education, technical skills, job role, company size, and location. Accurately estimating salary can help organizations make data-driven decisions during hiring, compensation planning, and workforce analysis.

This project builds a machine learning regression system that predicts employee salaries based on various employee attributes. Multiple regression algorithms are trained, evaluated, and compared to identify the best-performing model.

The final model can estimate an employee's expected salary using their professional profile information.

---

# Business Problem

Organizations often need accurate salary estimation while hiring new employees or reviewing compensation structures.

The objective of this project is:

> To develop a machine learning model that predicts employee salary based on employee experience, skills, education, role, and other relevant factors.

---

# Machine Learning Problem

This is a **Supervised Machine Learning Regression Problem**.

### Target Variable

```
salary_bdt
```

The model predicts the continuous salary value of an employee.

---

# Dataset Description

The dataset contains employee professional and demographic information.

## Features

### Numerical Features

| Feature                           | Description                     |
| --------------------------------- | ------------------------------- |
| age                               | Employee age                    |
| experience_years                  | Total years of experience       |
| skills_count                      | Number of technical skills      |
| certifications                    | Number of certifications        |
| worked_remote                     | Remote working status           |
| last_promotion_years_ago          | Time since last promotion       |
| recent_project_description_length | Recent project description size |


### Categorical Features

| Feature | Description |
|---|---|
| gender | Employee gender |
| education | Educational qualification |
| role_seniority | Job seniority level |
| company_size | Organization size |
| location_tier | Work location category |

### Target

```
salary_bdt
```

---

# Project Workflow

The project follows a complete machine learning workflow:

```
Data Collection
      |
Data Cleaning
      |
Exploratory Data Analysis
      |
Feature Engineering
      |
Data Preprocessing
      |
Model Training
      |
Model Evaluation
      |
Model Saving
      |
Salary Prediction
```

---

# Data Processing

## 1. Data Cleaning

The dataset was cleaned by performing:

- Dataset structure analysis
- Missing value detection
- Missing value handling
    - Numerical features → Median imputation
    - Categorical features → Mode imputation
- Duplicate record checking
- Duplicate removal
- Target variable analysis


---

## 2. Feature Engineering

Machine learning models require numerical inputs, therefore categorical features were transformed.

Techniques applied:

- Binary Encoding
    - Gender

- One-Hot Encoding
    - Education
    - Role Seniority
    - Company Size
    - Location Tier


Dataset transformation:

```
Before Encoding:
Rows    : 2000
Columns : 15


After Encoding:
Rows    : 2000
Columns : 27
```

---

# Machine Learning Models

The following regression algorithms were trained and compared:

1. Linear Regression

2. Decision Tree Regressor

3. Random Forest Regressor

4. XGBoost Regressor


---

# Model Evaluation

Since this is a regression problem, the following metrics were used:

## R² Score

Measures how well the model explains the variation in salary prediction.

Higher value indicates better performance.

---

## Mean Absolute Error (MAE)

Measures the average difference between actual and predicted salary.

Lower value indicates better performance.

---

## Root Mean Squared Error (RMSE)

Measures prediction error and penalizes larger mistakes.

Lower value indicates better performance.

---

| Model             | R² Score | MAE       | RMSE      |
| ----------------- | -------- | --------- | --------- |
| Linear Regression | 74.61%   | 12,828.81 | 28,290.80 |
| Decision Tree     | 38.15%   | 22,177.58 | 44,155.68 |
| Random Forest     | 70.21%   | 14,697.59 | 30,644.85 |
| XGBoost           | 69.90%   | 14,096.64 | 30,802.07 |


---

# Final Model Selection

## Selected Model: Linear Regression

Linear Regression achieved the highest R² Score among all evaluated models and provided the lowest prediction error.

| Metric   | Value         |
| -------- | ------------- |
| R² Score | 74.61%        |
| MAE      | 12,828.81 BDT |
| RMSE     | 28,290.80 BDT |


The model was selected as the final salary prediction model.

# Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-Learn
- XGBoost
- Joblib


---

# Installation and Usage

## Clone Repository

```bash
git clone <https://github.com/banoththarun99/ML_PROJECTS>
```

---

## Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Run Data Cleaning

```bash
python src/data_cleaning.py
```

---

## Run Feature Engineering

```bash
python src/feature_engineering.py
```

---

## Train Machine Learning Models

```bash
python src/train_model.py
```

---

## Evaluate Model

```bash
python src/evaluation.py
```

---

## Predict Salary

```bash
python src/prediction.py
```

---

# Model Output

The project generates:

```
models/
│
├── best_model.pkl
└── scaler.pkl
```

These files are used for making future salary predictions without retraining the model.

---

# Key Learnings

Through this project, the following machine learning concepts were implemented:

- Data preprocessing
- Missing value handling
- Feature engineering
- Categorical encoding
- Regression model development
- Model comparison
- Performance evaluation
- Model serialization using Joblib


---

# Future Improvements

Possible improvements:

- Hyperparameter tuning
- Advanced feature selection
- Additional employee attributes
- Deployment using Streamlit
- REST API development
- Cloud deployment using MLOps practices


---

# Author

**Banoth Tharun**

B.Tech - Artificial Intelligence and Data Science