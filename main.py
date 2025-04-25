from data_cleaning import DataCleaning
from model_training import ModelTrainer

# Step 1: Clean the data
data_cleaner = DataCleaning(file_path="Telco_customer_churn.xlsx")  # Provide path to your raw data
cleaned_data = data_cleaner.clean_data()
data_cleaner.save_cleaned_data("cleaned_churn_data.xlsx")

# Step 2: Train and evaluate models
model_trainer = ModelTrainer(cleaned_data_path="cleaned_churn_data.xlsx")
best_model = model_trainer.train_and_evaluate()

# Note: Run Streamlit separately
# You can now run the Streamlit app using: `streamlit run streamlit_app.py`
