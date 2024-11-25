import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE

def feature_engineering(df):
    """
    Apply feature engineering to enhance the dataset.
    """
    # Combine grade levels into broader categories
    df["Performance_Level"] = pd.cut(
        df["G3"],
        bins=[0, 9, 14, 20],
        labels=["Low", "Medium", "High"]
    )

    # Interaction feature: Study time multiplied by absences
    df["Study_Absence_Interaction"] = df["studytime"] * df["absences"]

    return df

def load_and_preprocess_data(filepath):
    """
    Load, clean, and preprocess the student performance dataset.
    """
    # Load dataset
    df = pd.read_csv(r"D:\Projects\student-performance-classification\data\student-mat.csv", sep=";")

    # Apply feature engineering
    df = feature_engineering(df)

    # Encode categorical variables
    categorical_features = df.select_dtypes(include=["object"]).columns
    label_encoders = {col: LabelEncoder() for col in categorical_features}
    for col in categorical_features:
        df[col] = label_encoders[col].fit_transform(df[col])

    # Define features and target
    X = df.drop(["G3", "Performance_Level"], axis=1)  # Exclude target variables
    y = df["G3"]  # Use G3 as the target grade

    # Normalize numerical features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Filter out classes with fewer than 5 samples
    min_samples = 5
    valid_classes = y.value_counts()[y.value_counts() >= min_samples].index
    filtered_indices = y.isin(valid_classes)
    X, y = X[filtered_indices], y[filtered_indices]

    # Handle class imbalance using SMOTE
    smote = SMOTE(random_state=42, k_neighbors=1)
    X, y = smote.fit_resample(X, y)

    # Split the data
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

if __name__ == "__main__":
    # Test the data processing pipeline
    data_path = "../data/student-mat.csv"  # Adjust path if needed
    try:
        X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

        print("Data processing completed successfully!")
        print(f"Training set size: {X_train.shape}")
        print(f"Test set size: {X_test.shape}")
        print(f"Target distribution in training set:\n{pd.Series(y_train).value_counts()}")
        print(f"Target distribution in test set:\n{pd.Series(y_test).value_counts()}")
    except Exception as e:
        print(f"Error during data processing: {e}")
