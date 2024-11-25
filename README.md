# Student Performance Classification

This project implements a machine learning pipeline to predict and classify student performance based on various demographic, behavioral, and academic features. By analyzing a dataset of secondary school students, the project provides insights into factors that influence academic success and builds predictive models to help identify students at risk of underperformance.

## 🧑‍🏫 Introduction

The goal of this project is to analyze and classify students' academic performance using machine learning. The target variable is the students' final grade, predicted based on features such as study time, absences, previous grades, and demographic information.

### Use Cases:
- Identifying students who need additional support.
- Helping educators design personalized teaching strategies.
- Providing actionable insights for parents and schools to improve student outcomes.

---

## 🌟 Features

1. **Exploratory Data Analysis (EDA):**
   - Understand relationships between features using heatmaps.
   - Visualize grade distributions to identify patterns in student performance.

2. **Data Preprocessing:**
   - Feature engineering to create meaningful new variables.
   - Handling class imbalance using **SMOTE** (Synthetic Minority Oversampling Technique).
   - Normalization and encoding of features for improved model performance.

3. **Machine Learning Models:**
   - Logistic Regression
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Hyperparameter optimization using **GridSearchCV**.

4. **Model Evaluation:**
   - Classification reports and weighted F1-scores.
   - Confusion matrices to analyze prediction accuracy for each grade level.

5. **Visualizations:**
   - Correlation heatmaps, grade distributions, and confusion matrices.

---

## 📊 Dataset Description

The dataset used is the **Student Performance Dataset** from the **UCI Machine Learning Repository**, originally introduced in the paper:

**"Predicting Student Performance: An Application of Data Mining Methods with Educational Data"**  
By: Paulo Cortez and Alice Silva (2012).

### Key Features:

1. **Demographic Features:**
   - `age`: Student's age.
   - `sex`: Gender (M/F).
   - `address`: Urban or rural home.
   - `famsize`: Family size (small/large).

2. **Parental Features:**
   - `Medu` and `Fedu`: Education levels of the mother and father.
   - `Mjob` and `Fjob`: Occupations of the mother and father.

3. **Academic Features:**
   - `studytime`: Weekly study time.
   - `failures`: Number of past class failures.
   - `G1`, `G2`, `G3`: Grades for first, second, and final periods.

4. **Behavioral Features:**
   - `goout`: Frequency of going out with friends.
   - `Dalc`, `Walc`: Alcohol consumption on weekdays and weekends.
   - `absences`: Total number of absences.

**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Student+Performance)

## 🔧 Setup and Installation

### Prerequisites
- Python 3.7 or higher
- pip (Python package manager)

### Installation Steps

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/student-performance-classification.git
   cd student-performance-classification
