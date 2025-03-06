🚀 ML Level 2 - Advanced Models (XGBoost, LightGBM, CatBoost)

A collection of high-performance machine learning models implemented in Python, focusing on gradient boosting techniques.






📌 Overview

This repository explores advanced machine learning models, focusing on gradient boosting techniques to solve classification and regression problems. Each model is implemented in Python and run in Google Colab.

📂 Project Structure

ML-Level-2/
├── classification/
│   ├── xgboost_classifier.ipynb
│   ├── lightgbm_classifier.ipynb
│   ├── catboost_classifier.ipynb
│   ├── data/
│   │   ├── churn_modelling.csv
├── regression/
│   ├── xgboost_regressor.ipynb
│   ├── lightgbm_regressor.ipynb
│   ├── catboost_regressor.ipynb
│   ├── regression_model_comparison.ipynb
│   ├── data/
│   │   ├── insurance.csv
├── README.md

📚 Models & Features

🔹 Classification Models

✔ XGBoost Classifier

✔ LightGBM Classifier

✔ CatBoost Classifier

🔹 Regression Models

✔ XGBoost Regressor

✔ LightGBM Regressor

✔ CatBoost Regressor

🔹 Model Comparison

📊 Side-by-side performance comparison

📌 Hyperparameter tuning insights

📈 Visualization of model evaluation metrics

🚀 Running the Models

1️⃣ Setup Environment

Make sure you have the necessary libraries installed:

!pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost

2️⃣ Run Classification Models

# Example: Running XGBoost Classifier
from xgboost import XGBClassifier

model = XGBClassifier()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

3️⃣ Run Regression Models

# Example: Running LightGBM Regressor
from lightgbm import LGBMRegressor

model = LGBMRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

📊 Results & Insights

Model

Accuracy (Classification)

RMSE (Regression)

XGBoost

95.2%

4.57

LightGBM

94.8%

4.72

CatBoost

96.1%

4.49

🚀 CatBoost showed the highest classification accuracy!📉 XGBoost performed best in regression!

📜 License

This project is released under the MIT License. Feel free to use and modify the code.

🙌 Acknowledgments

A huge thanks to Hadelin de Ponteves and Kirill Eremenko for their outstanding contributions to machine learning education! Their work in the SuperDataScience Machine Learning A-Z course has been invaluable in shaping this project. 🎉

🔥 Let's build powerful ML models together! If you find this repository useful, give it a ⭐ and fork it!

