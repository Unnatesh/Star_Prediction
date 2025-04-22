# 🌟 Star Prediction App

A machine learning web app that predicts whether a celestial object is a **Star**, **Galaxy**, or **Quasar** using astronomical data.

## 🚀 Live Demo

Check out the live app here:  
👉 [Star Prediction Streamlit App](https://starprediction-nhyrypuhaw5jw2amjvb3z4.streamlit.app)

## 📊 Features

- Interactive data visualizations using **Plotly** and **Altair**
- Multiple ML models (Random Forest, SVM, XGBoost, etc.)
- Evaluation metrics (Accuracy, ROC-AUC, Confusion Matrix)
- Outlier handling & SMOTE for imbalance
- Clean UI with Streamlit

## 🧠 Models Used

- Logistic Regression  
- Random Forest  
- Support Vector Machine  
- K-Nearest Neighbors  
- Naive Bayes  
- XGBoost  
- Gradient Boosting  

## 📁 Dataset

`star_classification.csv` — contains astrophysical features for object classification.

## 🛠️ Installation

Clone the repo and install dependencies:

```bash
git clone https://github.com/unnatesh/Star_Prediction.git
cd Star_Prediction
pip install -r requirements.txt
streamlit run stream.py
