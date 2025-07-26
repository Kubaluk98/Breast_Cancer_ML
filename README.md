# Breast Cancer Prediction using Logistic Regression (with K-Fold Cross Validation)

![Python](https://img.shields.io/badge/Python-3.x%2B-blue.svg)
![scikit-learn](https://img.shields.io/badge/scikit--learn-0.24.2%2B-orange.svg)
![Pandas](https://img.shields.io/badge/Pandas-1.3.4%2B-lightgrey.svg)
![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)

## Project Overview

This project demonstrates the application of **Logistic Regression**, a fundamental machine learning algorithm, to predict breast cancer malignancy. It utilizes various clinical features to classify tumors as either benign or malignant. To ensure robust model evaluation, **K-Fold Cross Validation** is also implemented.

## Dataset

The dataset used for this project is `breast_cancer.csv`. This dataset is commonly derived from the [Wisconsin Breast Cancer (Diagnostic) Dataset](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic) available from the UCI Machine Learning Repository.

It contains features computed from a digitized image of a fine needle aspirate (FNA) of a breast mass. These features describe characteristics of the cell nuclei present in the image.

**Key Columns:**
* **Features (X):** Columns representing various characteristics of the cell nuclei (e.g., `radius_mean`, `texture_mean`, `perimeter_mean`, `area_mean`, `smoothness_mean`, `compactness_mean`, `concavity_mean`, `concave points_mean`, `symmetry_mean`, `fractal_dimension_mean`, and their corresponding `_se` and `_worst` values). Your code uses `ds.iloc[:, 1:-1]`, which suggests it takes all columns from the second to the second-to-last as features.
* **Target (y):** The last column, typically `diagnosis`, which indicates the classification (M = Malignant, B = Benign).

**Note:** The `breast_cancer.csv` file should be placed in the root directory of this repository for the code to run correctly. If not included, please obtain it from the UCI ML Repository or a similar source.

Results

Confusion Matrix:
[[TN FP]
 [FN TP]]
 
[[84  3]
 [ 3 47]]

 
Accuracy Score: 0.9562043795620438

Accuracy: 96.70 %
Standard Deviation: 1.97 %

Libraries Used

    Pandas: A powerful library for data manipulation and analysis.

    scikit-learn: A comprehensive machine learning library for model building, evaluation, and data preprocessing.
