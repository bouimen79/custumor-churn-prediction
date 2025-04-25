import pandas as pd
from sklearn.preprocessing import LabelEncoder

class DataCleaning:
    def __init__(self, file_path):
        """
        Initializes the DataCleaning class with the file path to the dataset.
        :param file_path: Path to the raw dataset (Excel/CSV)
        """
        self.file_path = file_path
        self.df = pd.read_excel(self.file_path)  # Read the raw dataset
        self.columns_to_drop = [
            'CustomerID', 'Churn Label', 'Lat Long', 'City', 'State', 'Country', 'Zip Code'
        ]

    def drop_unneeded_columns(self):
        """
        Drops unnecessary columns that are not needed for the analysis.
        """
        self.df.drop(columns=self.columns_to_drop, inplace=True)

    def initial_exploration(self):
        """
        Prints initial exploration information of the dataset.
        """
        print("Shape:", self.df.shape)
        print("\nInfo:")
        print(self.df.info())
        print("\nMissing values:\n", self.df.isnull().sum())

    def handle_total_charges(self):
        """
        Converts 'Total Charges' to numeric and drops rows with missing values in 'Total Charges'.
        """
        self.df['Total Charges'] = pd.to_numeric(self.df['Total Charges'], errors='coerce')
        self.df = self.df.dropna(subset=['Total Charges'])  # Drop rows where 'Total Charges' is NaN

    def encode_categorical_columns(self):
        """
        Encodes categorical columns (binary: LabelEncoding, multi-category: OneHotEncoding).
        """
        # Convert object columns to 'category' dtype
        for col in self.df.select_dtypes('object').columns:
            self.df[col] = self.df[col].astype('category')

        # Binary encoding for 2-category columns
        le = LabelEncoder()
        binary_cols = [col for col in self.df.select_dtypes('category').columns if self.df[col].nunique() == 2]

        for col in binary_cols:
            self.df[col] = le.fit_transform(self.df[col])

        # One-hot encoding for multi-category columns
        multi_cat_cols = [col for col in self.df.select_dtypes('category').columns if self.df[col].nunique() > 2]
        self.df = pd.get_dummies(self.df, columns=multi_cat_cols)

    def final_check(self):
        """
        Prints final check details after data cleaning.
        """
        print("\nFinal dataset shape:", self.df.shape)
        print("\nColumn types:\n", self.df.dtypes.value_counts())
        print("\nSample of dataset:\n", self.df.head())

    def clean_data(self):
        """
        Executes the full data cleaning pipeline.
        """
        self.drop_unneeded_columns()
        self.initial_exploration()
        self.handle_total_charges()
        self.encode_categorical_columns()
        self.final_check()
     
    def save_cleaned_data(self, output_path):
        # Save the cleaned DataFrame as an Excel file
        self.df.to_excel(output_path, index=False)
        print(f"Cleaned data saved to {output_path}")
        
        return self.df  # Return the cleaned dataframe


