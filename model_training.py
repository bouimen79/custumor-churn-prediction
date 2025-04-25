import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split

class ModelTrainer:
    def __init__(self, cleaned_data_path):
        """
        Initializes the ModelTrainer class with the cleaned dataset.
        :param cleaned_data_path: Path to the cleaned dataset (Excel/CSV).
        """
        self.df = pd.read_excel(cleaned_data_path)  # Read the cleaned data
        self.X = self.df.drop("Churn Value", axis=1)  # Features
        self.y = self.df["Churn Value"]  # Target (Churn Value)

    def train_and_evaluate(self):
        """
        Trains and evaluates different models (Logistic Regression and Random Forest).
        It selects the best model based on the F1 score and saves it.
        """
        # Split the dataset into training and test sets (80% train, 20% test)
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        # Dictionary of models to evaluate
        models = {
            "Logistic Regression": LogisticRegression(max_iter=1000),
            "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42,class_weight='balanced')
        }

        best_model = None
        best_score = 0
        best_name = ""

        # Train and evaluate each model
        for name, model in models.items():
            model.fit(X_train, y_train)  # Train the model
            y_pred = model.predict(X_test)  # Predict on the test set

            # Evaluate the model using F1 score
            f1 = f1_score(y_test, y_pred)
            if f1 > best_score:
                best_model = model
                best_score = f1
                best_name = name

            # Print evaluation metrics for each model
            print(f"{name} F1-score: {f1:.2%}")

        # Save the best model and the column names used for training
        joblib.dump(best_model, "churn_model.pkl")
        joblib.dump(self.X.columns.tolist(), "model_columns.pkl")

        print(f"Best Model: {best_name} with F1-score: {best_score:.2%}")
        return best_model  # Return the best model

