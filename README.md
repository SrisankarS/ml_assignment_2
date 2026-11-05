# Dry Bean Classification Using Machine Learning

## Name: Sri Sankar Sivakumar
## BITS ID: 2026AK05269
## ML Assignment: 2

## 1. Problem Statement

The objective of this project is to develop and compare five supervised machine learning classification algorithms to classify dry bean varieties using numerical features extracted from bean images.

The five classification algorithms used are:

1. Logistic Regression
2. Decision Tree
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

The models are evaluated using Accuracy, AUC, Precision, Recall, F1 Score and Matthews Correlation Coefficient (MCC).

## 2. Dataset Description

**Dataset:** Dry Bean Dataset  
**Source:** UCI Machine Learning Repository  
**Dataset URL:** https://archive.ics.uci.edu/dataset/602/dry+bean

Dataset characteristics:

- Total instances: 13,611
- Input features: 16 numerical features
- Target variable: Class
- Number of classes: 7
- Classification type: Multiclass classification

The seven bean classes are:

- SEKER
- BARBUNYA
- BOMBAY
- CALI
- DERMASON
- HOROZ
- SIRA

The dataset was divided into training and testing sets using an 80:20 split with stratification.

## 3. Methodology

The following steps were performed:

1. Loaded and inspected the dataset.
2. Performed exploratory data analysis.
3. Separated input features and target variable.
4. Split the dataset into training and testing sets.
5. Applied feature standardization where appropriate.
6. Trained five classification algorithms.
7. Evaluated all models using six performance metrics.
8. Saved the trained models using Joblib.

For Logistic Regression, KNN and SVM, StandardScaler was included in the model pipeline. Decision Tree and Gaussian Naive Bayes were trained without feature scaling.

For multiclass evaluation, macro averaging was used for Precision, Recall and F1 Score. AUC was calculated using the One-vs-Rest (OvR) strategy with macro averaging.

## 4. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.921410 | 0.994776 | 0.935383 | 0.932149 | 0.933538 | 0.905045 |
| Decision Tree | 0.892031 | 0.944996 | 0.907513 | 0.909028 | 0.908061 | 0.869569 |
| KNN | 0.916636 | 0.983317 | 0.932026 | 0.927059 | 0.929259 | 0.899204 |
| Gaussian Naive Bayes | 0.763863 | 0.967193 | 0.774427 | 0.769417 | 0.767750 | 0.715406 |
| Support Vector Machine | **0.922145** | **0.994927** | **0.935937** | **0.932952** | **0.934297** | **0.905870** |

## 5. Model Observations

### Logistic Regression

Logistic Regression achieved an accuracy of 92.14% and an AUC of 0.994776. It demonstrated strong classification performance and was a close competitor to SVM across all evaluation metrics.

### Decision Tree

Decision Tree achieved an accuracy of 89.20%. Its performance was lower than Logistic Regression, KNN and SVM in this experiment.

### K-Nearest Neighbors

KNN achieved an accuracy of 91.66% and an AUC of 0.983317. It performed well but was slightly behind Logistic Regression and SVM.

### Gaussian Naive Bayes

Gaussian Naive Bayes achieved an accuracy of 76.39%. It had the lowest accuracy, precision, recall, F1 Score and MCC among the five evaluated models.

### Support Vector Machine

SVM achieved the highest accuracy of 92.21%, the highest AUC of 0.994927 and the highest values across all six evaluation metrics. It was the best-performing classifier in this experiment.

## 6. Overall Winner

Support Vector Machine (SVM) was the overall best-performing model based on the six evaluation metrics.

It achieved:

- Accuracy: 0.922145
- AUC: 0.994927
- Precision: 0.935937
- Recall: 0.932952
- F1 Score: 0.934297
- MCC: 0.905870

Logistic Regression was a close second, with very similar performance.

## 7. Project Structure

```text
ML_Assignment_2/
│
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── model_evaluation_results.csv
│
└── models/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```

## 8. Application

The Streamlit application allows users to upload a labelled test CSV file, select one of the five saved classification models, and view evaluation metrics, a
