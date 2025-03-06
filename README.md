# 🚀🔥 ML Classics in Python - Level 2 (Google Colab) 🔥 
**Advanced Machine Learning models implemented in Python using Google Colab**  

![ML](https://img.shields.io/badge/Machine_Learning-Python-blue) ![Boosting](https://img.shields.io/badge/Boosting-XGBoost%20%7C%20LightGBM%20%7C%20CatBoost-orange) ![Status](https://img.shields.io/badge/Status-Active-green) ![License](https://img.shields.io/badge/License-MIT-lightgrey)  

---

## 📌 Description  
🚀 This repository provides implementations of **advanced machine learning models** in **Python**, structured into key parts. It is designed for those who want to **deepen their ML knowledge** beyond basic models. Here, you will explore **regression, classification, boosting algorithms, and model comparison**.

### 🏆 Why this project?  
✅ **Hands-on Learning** – Practical implementation of advanced ML models  
✅ **Structured Approach** – Organized into key ML categories for easy navigation  
✅ **Boosting & Model Comparisons** – Learn powerful ML techniques  
✅ **Educational Purpose** – Inspired by the SuperDataScience ML A-Z course  

---

## 📂 Project Structure 📁  
```bash
ML-Classics-Level2/
├── Part 1 - Advanced Regression/      # Advanced regression models
├── Part 2 - Advanced Classification/  # Advanced classification models
├── Part 3 - Model Comparison/         # Comparing ML models
├── data/                              # Datasets
├── README.md                          # Documentation
```

---

## 📚 Content  
### 🟢 **Part 1: Advanced Regression**  
✔ **CatBoost Regressor** [`(S1) catboost_regressor.ipynb`]  
✔ **LightGBM Regressor** [`(S1) lightgbm_regressor.ipynb`]  
✔ **XGBoost Regressor** [`(S1) xgboost_regressor.ipynb`]  

### 🔵 **Part 2: Advanced Classification**  
✔ **CatBoost Classifier** [`(S1) catboost_classifier.ipynb`]  
✔ **LightGBM Classifier** [`(S1) lightgbm_classifier.ipynb`]  
✔ **XGBoost Classifier** [`(S1) xgboost_classifier.ipynb`]  



### 🎯 **Part 3: Model Comparison**  
✔ **Regression Model Comparison** [`(S1) regression_model_comparison.ipynb`]  

---

## 📊 Regression Model Performance  
| Model               | MAE    | MSE        | RMSE    | R² Score | Training Time |
|--------------------|--------|------------|--------|----------|--------------|
| **CatBoost**       | 2494.64 | 1.858e+07  | 4310.46 | 0.8803   | 0.0612       |
| **XGBoost**        | 2466.08 | 1.813e+07  | 4258.64 | 0.8832   | 0.2022       |
| **LightGBM**       | 2471.82 | 1.834e+07  | 4283.38 | 0.8818   | 0.1180       |

---

## 📊 Classification Model Performance  
| Model               | Accuracy | Precision | Recall | F1 Score | ROC AUC | Training Time |
|--------------------|----------|-----------|--------|----------|--------|--------------|
| **Voting Classifier** | 0.8705  | 0.8033    | 0.4816 | 0.6022   | 0.7257 | N/A          |
| **CatBoost**       | 0.8700  | 0.7905    | 0.4914 | 0.6061   | 0.7291 | 8.0106       |
| **LightGBM**       | 0.8685  | 0.7812    | 0.4914 | 0.6033   | 0.7281 | 0.3591       |
| **XGBoost**        | 0.8530  | 0.6969    | 0.4914 | 0.5764   | 0.7184 | 5.1836       |

---

## 🚀 How to Use?  
### 🔧 Installation  
Ensure you have the required libraries installed before running the scripts:  
```python
!pip install numpy pandas matplotlib seaborn scikit-learn xgboost lightgbm catboost
```

### ▶ Running the Scripts  
1️⃣ **Clone the repository**:  
```bash
git clone https://github.com/evgeniimatveev/Ml-Classics-Level2-Boosting.git
cd Ml-Classics-Level2-Boosting
```

2️⃣ **Run the scripts in Google Colab**:  
```python
from google.colab import drive
drive.mount('/content/drive')
```

3️⃣ **Open the required notebook**:  
```python
%cd '/content/drive/My Drive/Colab Notebooks/Part 1 - Advanced Regression'
!jupyter notebook (S1) catboost_regressor.ipynb
```

---

## 📌 Authors & Acknowledgments  
🔹 **Developed by:** **Evgenii Matveev**  
🔹 **Source:** **SuperDataScience Machine Learning A-Z (Python)**  
🔹 **For educational purposes only**  

🔥 **Special thanks** to the original authors of the SuperDataScience course – **Hadelin de Ponteves** and **Kirill Eremenko** for their contributions to ML education! 🚀🙌  

---

## 📜 License  
This project is distributed under the **MIT License**. Feel free to use the code! 🚀  

---

## 📢 Stay Connected!  
💻 **GitHub Repository:** [ML-Classics-Level2-Boosting](https://github.com/evgeniimatveev/Ml-Classics-Level2-Boosting)  
🌐 **Portfolio:** [Data Science Portfolio](https://www.datascienceportfol.io/evgeniimatveevusa)  
📌 **LinkedIn:** [Evgenii Matveev](https://www.linkedin.com/in/evgenii-matveev-510926276/)  


---

🔥 **If you like this project, don't forget to star ⭐ the repository!** 🔥
