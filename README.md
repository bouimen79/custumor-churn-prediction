# custumor-churn-prediction
A complete machine learning project to predict customer churn using real-world telecom data. This project includes data cleaning, model training, evaluation, and a deployable Streamlit web application for live predictions.
# Project Features
✅ End-to-end machine learning pipeline
🧹 Data cleaning & feature engineering
📊 Handling categorical data & class imbalance
🤖 Model training with Logistic Regression & Random Forest
🧠 Evaluation using F1-score
🌐 Streamlit app for real-time customer churn prediction
🗂️ Saved model & columns using joblib

# Technologies Used
Python
Pandas, NumPy
Scikit-learn
Streamlit
Joblib
#  How to Run
## 1. Clone the repo
git clone https://github.com/your-username/churn-prediction-app.git
cd churn-prediction-app
## 2. Install dependencies
pip install -r requirements.txt
## clean data 
from data_cleaner import DataCleaner

cleaner = DataCleaner("your_dataset.xlsx")
df_cleaned = cleaner.clean()
df_cleaned.to_excel("cleaned_data.xlsx", index=False)
## Train model
from model_trainer import ModelTrainer

trainer = ModelTrainer("cleaned_data.xlsx")
model = trainer.train_and_evaluate()
## Run Streamlit app 
streamlit run streamlit_app.py

