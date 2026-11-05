# Creating a python file to fed into Streamlit Web Application

import os
import joblib
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

# --------------------------------------------------
# 1. Page configuration
# --------------------------------------------------

st.set_page_config(
    page_title="Dry Bean ML Model Evaluation",
    page_icon="🌱",
    layout="wide"
)

st.title("🌱 Dry Bean Classification")
st.subheader("Machine Learning Model Evaluation")
st.markdown("""
<style>
div[data-testid="stMetricValue"] {
    font-size: 1.5rem;
}
div[data-testid="stMetricLabel"] {
    font-size: 0.85rem;
}
</style>
""", unsafe_allow_html=True)

st.write(
    "Upload a labelled test dataset, select a trained "
    "classification model, and evaluate its performance."
)

# --------------------------------------------------
# 2. Expected dataset columns
# --------------------------------------------------

feature_columns = [
    "Area",
    "Perimeter",
    "MajorAxisLength",
    "MinorAxisLength",
    "AspectRation",
    "Eccentricity",
    "ConvexArea",
    "EquivDiameter",
    "Extent",
    "Solidity",
    "roundness",
    "Compactness",
    "ShapeFactor1",
    "ShapeFactor2",
    "ShapeFactor3",
    "ShapeFactor4"
]

target_column = "Class"

# --------------------------------------------------
# 3. Model file locations
# --------------------------------------------------

model_files = {
    "Logistic Regression": "models/logistic_regression.joblib",
    "Decision Tree": "models/decision_tree.joblib",
    "KNN": "models/knn.joblib",
    "Naive Bayes": "models/naive_bayes.joblib",
    "Support Vector Machine": "models/svm.joblib"
}

# --------------------------------------------------
# 4. Load selected trained model
# --------------------------------------------------

@st.cache_resource
def load_model(model_path):
    return joblib.load(model_path)


# --------------------------------------------------
# 5. Upload test dataset
# --------------------------------------------------

uploaded_file = st.file_uploader(
    "Upload your test CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        test_df = pd.read_csv(uploaded_file)

        st.success("CSV file uploaded successfully!")

        st.write("### Dataset preview")
        st.dataframe(test_df.head(), use_container_width=True)

        # Check required columns
        required_columns = feature_columns + [target_column]

        missing_columns = [
            col for col in required_columns
            if col not in test_df.columns
        ]

        if missing_columns:
            st.error(f"Missing required columns: {missing_columns}")
            st.stop()

        if test_df.empty:
            st.error("The uploaded CSV contains no records.")
            st.stop()

        X_test = test_df[feature_columns]
        y_test = test_df[target_column]

        if X_test.isnull().any().any() or y_test.isnull().any():
            st.error(
                "The uploaded dataset contains missing values. "
                "Please upload a complete test dataset."
            )
            st.stop()

        if not all(
            pd.api.types.is_numeric_dtype(X_test[col])
            for col in feature_columns
        ):
            st.error("All 16 input features must be numeric.")
            st.stop()

        st.write("### Dataset information")
        st.write("Number of records:", len(test_df))
        st.write("Number of input features:", len(feature_columns))
        st.write("Number of classes:", y_test.nunique())

        # --------------------------------------------------
        # 6. Select model
        # --------------------------------------------------

        st.write("### Select a classification model")

        selected_model_name = st.selectbox(
            "Choose your model",
            list(model_files.keys())
        )

        # --------------------------------------------------
        # 7. Load model and generate predictions
        # --------------------------------------------------

        model_path = model_files[selected_model_name]

        if not os.path.exists(model_path):
            st.error(f"Model file not found: {model_path}")
            st.stop()

        model = load_model(model_path)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # --------------------------------------------------
        # 8. Calculate the 6 evaluation metrics
        # --------------------------------------------------

        class_labels = model.classes_

        if set(y_test.unique()) != set(class_labels):
            st.error(
                "The uploaded test dataset must contain all seven "
                "bean classes represented by the selected trained model "
                "to calculate multiclass AUC consistently."
            )
            st.stop()

        accuracy = accuracy_score(y_test, y_pred)

        auc = roc_auc_score(
            y_test,
            y_proba,
            labels=class_labels,
            multi_class="ovr",
            average="macro"
        )

        precision = precision_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            y_pred,
            average="macro",
            zero_division=0
        )

        mcc = matthews_corrcoef(y_test, y_pred)

        # --------------------------------------------------
        # 9. Display evaluation metrics
        # --------------------------------------------------

        st.write("### Model Evaluation Results")
        st.write("Selected model:", selected_model_name)

        col1, col2, col3 = st.columns(3)

        col1.metric("Accuracy", f"{accuracy:.6f}")
        col2.metric("AUC", f"{auc:.6f}")
        col3.metric("Precision", f"{precision:.6f}")

        col4, col5, col6 = st.columns(3)

        col4.metric("Recall", f"{recall:.6f}")
        col5.metric("F1 Score", f"{f1:.6f}")
        col6.metric("MCC", f"{mcc:.6f}")

        # --------------------------------------------------
        # 10. Display confusion matrix
        # --------------------------------------------------

        st.write("### Confusion Matrix")

        cm = confusion_matrix(
            y_test,
            y_pred,
            labels=class_labels
        )

        fig, ax = plt.subplots(figsize=(5, 5), dpi=100)

        display = ConfusionMatrixDisplay(
            confusion_matrix=cm,
            display_labels=class_labels
        )

        display.plot(
            ax=ax,
            cmap="Blues",
            xticks_rotation=45,
            colorbar=False,
            values_format="d"
        )

        ax.set_title(
            f"Confusion Matrix - {selected_model_name}",
            fontsize=11
        )
        ax.set_xlabel("Predicted Class", fontsize=9)
        ax.set_ylabel("Actual Class", fontsize=9)
        ax.tick_params(axis="both", labelsize=8)
        ax.set_aspect("equal")

        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        plt.close(fig)

        # --------------------------------------------------
        # 11. Display classification report
        # --------------------------------------------------

        st.write("### Classification Report")

        report = classification_report(
            y_test,
            y_pred,
            labels=class_labels,
            target_names=[str(label) for label in class_labels],
            output_dict=True,
            zero_division=0
        )

        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df.round(4), use_container_width=True)

        # --------------------------------------------------
        # 12. Display predictions
        # --------------------------------------------------

        st.write("### Actual vs Predicted Classes")

        prediction_df = pd.DataFrame({
            "Actual Class": y_test.values,
            "Predicted Class": y_pred,
            "Correct Prediction": y_test.values == y_pred
        })

        st.dataframe(
            prediction_df.head(20),
            use_container_width=True
        )

        st.success("Model evaluation completed successfully!")

    except Exception as error:
        st.error(f"An error occurred: {error}")
