# Employee Attrition Prediction and Classification

An end-to-end HR analytics project for employee attrition prediction using exploratory data analysis, missing data simulation, PCA, logistic regression models, SMOTE, and cost-sensitive learning.


## Project Overview
This project analyzes employee attrition patterns using statistical modeling and machine learning techniques. The main objective is to identify important factors associated with employee turnover and to build interpretable classification models while handling real-world data challenges such as missing data and class imbalance.

The study follows a complete data science pipeline including exploratory analysis, confirmatory statistical testing, feature engineering, dimensionality reduction, and predictive modeling.


## Objectives

- Analyze factors affecting employee attrition  
- Explore relationships between demographic and job-related variables  
- Simulate and handle missing data  
- Apply dimensionality reduction techniques  
- Build predictive and interpretable classification models  
- Improve minority class detection performance  


## Dataset Information

- Source: IBM HR Analytics Dataset (Kaggle)  
- Number of observations: 1470  
- Number of features: 35  
- Target variable: Attrition (Yes / No)  

The dataset includes demographic information, job characteristics, performance indicators, and work-life balance attributes.


## Methodology

### Exploratory Data Analysis (EDA)

- Distribution analysis of numerical and categorical variables  
- Missing value structure visualization  
- Bivariate analysis between attrition and key variables  
- Correlation analysis among experience-related features  

Main observations:
- Employees with lower income and younger age tend to leave more frequently  
- Overtime shows a strong association with attrition  


### Missing Data Simulation and Imputation

- Synthetic missing values generated using MCAR mechanism  
- Approximately 7% missingness introduced for selected variables  
- Distribution preservation verified using statistical tests  

Imputation strategy:
- Numerical variables: Median imputation  
- Categorical variables: Most frequent value imputation  


### Confirmatory Statistical Analysis

- Normality tested using Shapiro-Wilk test and Q-Q plots  
- Homogeneity of variance tested using Levene’s test  

Due to assumption violations, non-parametric testing was applied:

- Mann–Whitney U Test used for group comparisons  

Significant differences were observed for:
- Monthly income  
- Age  
- Total working years  


### Feature Engineering and Dimensionality Reduction

- PCA applied to selected numerical features  
- First two principal components explained approximately 86% of total variance  
- PCA used for model comparison and visualization  

Additional visualization techniques:
- t-SNE  
- UMAP  

Both methods showed strong overlap between attrition classes, indicating complex decision boundaries.


## Predictive Modeling

### Experimental Setup

- Stratified train-test split  
- 5-fold stratified cross-validation  
- Pipeline-based preprocessing to prevent data leakage  

Preprocessing steps:
- Standardization  
- One-hot encoding  
- Imputation inside pipelines  


### Models Implemented

- Logistic Regression (Baseline)  
- Ridge Regression (L2)  
- Lasso Regression (L1)  
- Elastic Net  
- PCA-based Logistic Regression  
- SMOTE + Logistic Regression  
- Cost-sensitive Logistic Regression  


## Model Evaluation Metrics

Models were evaluated using:

- ROC-AUC  
- Sensitivity (Recall)  
- Specificity  
- F1-Score  
- Cohen’s Kappa  

## Results and Findings

- Baseline and regularized models achieved high ROC-AUC but low sensitivity  
- PCA-based model failed to detect minority class  
- SMOTE improved recall but reduced specificity  
- Cost-sensitive logistic regression provided the best balance  

### Final Selected Model

Cost-sensitive Logistic Regression achieved:

- Highest sensitivity  
- Best F1-score  
- Balanced performance without synthetic data generation  


## Important Attrition Factors

Higher attrition risk associated with:
- Frequent business travel  
- Overtime work  
- Long time since last promotion  
- Distance from home  

Lower attrition risk associated with:
- Higher monthly income  
- Higher job satisfaction  
- Better work-life balance  
- Longer tenure with current manager  


## Technologies Used

- Python  
- Pandas, NumPy  
- Scikit-learn  
- Imbalanced-learn  
- Matplotlib, Seaborn  

## Authors

- Başak Kabaloğlu  
- Fitnat Koç  
- Sabahattin Alp Kocabaş  

