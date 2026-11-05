# Dry Bean Classification Using Machine Learning

### Name: Sri Sankar Sivakumar
### BITS ID: 2026AK05269
### ML Assignment: 2

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

## 4. Github Repository Link:

## 5. Streamlit Application Link:     

## 6. Methodology

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

## 7. Model Comparison

| Model | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.921410 | 0.994776 | 0.935383 | 0.932149 | 0.933538 | 0.905045 |
| Decision Tree | 0.892031 | 0.944996 | 0.907513 | 0.909028 | 0.908061 | 0.869569 |
| KNN | 0.916636 | 0.983317 | 0.932026 | 0.927059 | 0.929259 | 0.899204 |
| Gaussian Naive Bayes | 0.763863 | 0.967193 | 0.774427 | 0.769417 | 0.767750 | 0.715406 |
| Support Vector Machine | **0.922145** | **0.994927** | **0.935937** | **0.932952** | **0.934297** | **0.905870** |

## 8. Model Observations

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Achieved 92.14% accuracy and 0.994776 AUC. Demonstrated strong classification performance, with a Macro F1 Score of 0.933538 and MCC of 0.905045. It was a close competitor to SVM across all six evaluation metrics. |
| Decision Tree | Achieved 89.20% accuracy and 0.944996 AUC. Its performance was lower than Logistic Regression, KNN and SVM in this experiment, with an MCC of 0.869569. |
| KNN | Achieved 91.66% accuracy and 0.983317 AUC. Demonstrated good classification performance, with a Macro F1 Score of 0.929259. It performed slightly below Logistic Regression and SVM. |
| Naive Bayes (Gaussian) | Achieved 76.39% accuracy and 0.967193 AUC. It recorded the lowest Accuracy, Precision, Recall, F1 Score and MCC among the five models, with an MCC of 0.715406. |
| Support Vector Machine | Achieved the highest accuracy of 92.21% and the highest AUC of 0.994927. It also achieved the highest Precision, Recall, F1 Score and MCC, making it the best-performing classifier in this experiment. |
| Overall, Winner for your dataset? | Support Vector Machine (SVM) was the overall winner, achieving the highest values across all six evaluation metrics. Logistic Regression was a close second, with very similar performance. |

## 9. Project Structure

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

## 10. Application

The Streamlit application allows users to upload a labelled test CSV file, select one of the five saved classification models, and view evaluation metrics, a confusion matrix and a classification report.

The application loads the previously trained models and does not retrain them.

