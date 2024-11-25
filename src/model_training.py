from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
import os

def optimize_model(model, param_grid, X_train, y_train):
    """
    Perform hyperparameter optimization using GridSearchCV.
    """
    grid_search = GridSearchCV(model, param_grid, scoring='f1_weighted', cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def train_and_evaluate_models(X_train, X_test, y_train, y_test):
    """
    Train multiple models, evaluate their performance, and save confusion matrices.
    """
    # Ensure output directory exists
    output_dir = "outputs/visualizations"
    os.makedirs(output_dir, exist_ok=True)

    # Define models and their hyperparameter grids
    models = {
        "Logistic Regression": (LogisticRegression(max_iter=1000), {"C": [0.1, 1, 10]}),
        "Random Forest": (RandomForestClassifier(), {"n_estimators": [50, 100], "max_depth": [10, 20]}),
        "Support Vector Machine": (SVC(), {"C": [0.1, 1], "kernel": ["linear", "rbf"]}),
    }

    for name, (model, param_grid) in models.items():
        print(f"Training {name}...")

        # Hyperparameter tuning
        best_model = optimize_model(model, param_grid, X_train, y_train)

        # Predictions and evaluation
        predictions = best_model.predict(X_test)
        print(f"\n{name} - Classification Report:\n")
        print(classification_report(y_test, predictions))

        # Confusion matrix
        cm = confusion_matrix(y_test, predictions)
        plt.figure(figsize=(6, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
        plt.title(f"{name} - Confusion Matrix")
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.savefig(f"{output_dir}/{name}_confusion_matrix.png")
        plt.show()

        # F1 Score
        weighted_f1 = f1_score(y_test, predictions, average="weighted")
        print(f"{name} - Weighted F1 Score: {weighted_f1:.2f}\n")
