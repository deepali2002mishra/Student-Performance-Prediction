import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from src.data_processing import load_and_preprocess_data
from src.model_training import train_and_evaluate_models

def exploratory_analysis(filepath):
    """
    Perform exploratory data analysis and save visualizations.
    """
    # Load dataset
    df = pd.read_csv(filepath, sep=";")

    # Correlation heatmap
    plt.figure(figsize=(12, 8))
    numeric_cols = df.select_dtypes(include=['number']).columns
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig("outputs/visualizations/correlation_heatmap.png")
    plt.close()

    # Distribution of final grades
    plt.figure(figsize=(8, 6))
    sns.histplot(df["G3"], kde=True, bins=20)
    plt.title("Distribution of Final Grades (G3)")
    plt.savefig("outputs/visualizations/grade_distribution.png")
    plt.close()

    print("Exploratory analysis completed. Visualizations saved.")

if __name__ == "__main__":
    # Filepath to dataset
    data_path = "data/student-mat.csv"

    # Step 1: Exploratory Analysis
    print("Running exploratory analysis...")
    exploratory_analysis(data_path)

    # Step 2: Data Preprocessing
    print("Preprocessing data...")
    X_train, X_test, y_train, y_test = load_and_preprocess_data(data_path)

    # Step 3: Model Training and Evaluation
    print("Training models...")
    train_and_evaluate_models(X_train, X_test, y_train, y_test)

    print("All steps completed successfully!")
