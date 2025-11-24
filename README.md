## 🏦 Bank Customer Churn Prediction

A complete end-to-end machine learning project that predicts whether a customer will leave a bank (churn) based on demographic, behavioral, and financial features.This project covers data preprocessing, feature engineering, EDA, model building, model comparison, and performance evaluation using Python and Scikit-Learn.

------------------------------------------------------------------------

## 📘 Project Overview

Customer churn is a major challenge for banks. Predicting which customers are likely to leave enables proactive retention strategies.This project builds an ML classification model that predicts customer churn based on factors like:

-  Credit score
-  Age
-  Tenure
-  Account balance
-  Number of products
-  Geography
-  Gender
-  Estimated salary

------------------------------------------------------------------------

## 📂 Dataset
The notebook loads a dataset with typical bank customer attributes:
```
   Feature            Description                     
 -----------------  --------------------------------- 
  `CreditScore`       Customer credit score            
  `Geography`         Country (France/Germany/Spain)    
  `Gender`            Male/Female                      
  `Age`               Customer age                     
  `Tenure`            Number of years with the bank    
  `Balance`           Account balance                 
  `NumOfProducts`     Number of products held           
  `HasCrCard`         Credit card ownership          
  `IsActiveMember`    Activity status                 
  `EstimatedSalary`   Salary estimate                  
  `Exited`            **Target** – churn (1) or not (0) 

```
------------------------------------------------------------------------

## 🧹 Data Preprocessing
✔ Steps Performed
-  Handling missing values (if any)
-  Encoding categorical variables:
   -  One-hot encoding for Geography
   -  Label encoding / binary conversion for Gender
- Feature scaling using:
   - StandardScaler for numerical features
-  Splitting dataset into:
   -  Training set
   -  Test set
------------------------------------------------------------------------

## 🧰 Libraries Used
   -   numpy
   -   pandas
   -   matplotlib
   -   seaborn
   -   scikit-learn
------------------------------------------------------------------------

## 🧠 Machine Learning Models Used
Your notebook trains multiple ML models such as:
-  Logistic Regression
-  Decision Tree Classifier
-  Random Forest Classifier
-  Support Vector Machine (SVM)
-  KNN Classifier
-  XGBoost / Gradient Boosting (if used in code)
  
Each model is fit on the training data and evaluated on the test data.

------------------------------------------------------------------------

## 📊 Model Evaluation Metrics
Performance is measured using:
-  Accuracy Score
-  Confusion Matrix
-  Classification Report (Precision, Recall, F1-Score)
-  ROC-AUC Score

Visualization tools include:
-  Heatmaps for confusion matrix
-  Feature importance plots (for tree-based models)

------------------------------------------------------------------------

## 🏆 Best Performing Model
Based on standard churn datasets, Random Forest and Gradient Boosting/XGBoost typically perform the best due to their ability to capture nonlinear patterns.

------------------------------------------------------------------------

## ▶️ How to Run

1️⃣ Install dependencies
```
pip install numpy pandas matplotlib seaborn scikit-learn xgboost

```
2️⃣ Open Jupyter Notebook
```
jupyter notebook "Bank Customer Churn Prediction.ipynb"

```
3️⃣ Run all cells to train and evaluate the model.

------------------------------------------------------------------------
## 🔍 Example ML Pipeline
```
# Scaling features
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Splitting dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)

# Random Forest model
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
```
------------------------------------------------------------------------
## 🌟 Future Improvements
-  Deploy model using Flask, FastAPI, or Streamlit
-  Add SHAP explainability
-  Balance dataset using SMOTE
-  Train advanced models:
    -  LightGBM
    -  CatBoost
    -  Stacking classifiers

-  Save pipeline using Pickle for production
  
-----------------------------------------------------------------------
## 👩‍💻 Author

**Monika A.D**
• AI & DS Student
(2025)

------------------------------------------------------------------------

Enjoy learning  🚀
