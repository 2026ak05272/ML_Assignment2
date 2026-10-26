# Telecom Customer Churn Classification

## a. Problem statement

Predict whether a telecom customer will leave the service (`Churn = Yes`) or remain (`Churn = No`) from their account, service, contract, and billing details. The project trains five classification algorithms, compares their held-out performance, and provides an interactive Streamlit interface for evaluating labeled test CSV files.

## b. Dataset description

- **Dataset:** Telco Customer Churn (the supplied `Telecom_customer_churn.csv`).
- **Original source:** [Kaggle — Telecom Customer Churn](https://www.kaggle.com/datasets/alipakzad/telecom-customer-churn).
- **Source in this workspace:** `Sample Data/Classification/Telecome Customer Churn/Telecom_customer_churn.csv`.
- **Rows:** 7,043 customer records.
- **Predictors:** 19 features, covering demographics, subscribed services, contract/payment preferences, tenure, and charges.
- **Target:** `Churn`, a binary Yes/No label (1,869 Yes; 5,174 No).
- **Data preparation:** The supplied data has 11 blank `TotalCharges` values. Numeric missing values are median-imputed and categorical missing values are most-frequent-imputed within each model pipeline. Categorical inputs are one-hot encoded; numeric inputs are standardized. Preprocessing is fit on training data only.
- **Split:** Stratified 80/20 train/test split with random state 42. The labeled test set is saved at `test_data.csv` for reproducibility and app demos.

## c. GitHub Repository Link

**Repository:** [https://github.com/2026ak05272/ML_Assignment2](https://github.com/2026ak05272/ML_Assignment2)

## d. Models used

The scores below are from the reproducible stratified 80/20 split (random state 42) and are saved in `model/metadata.json`. Metrics use `Yes` as the positive class. AUC is calculated from positive-class probabilities where available and decision scores for the SVM.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.738 | 0.841 | 0.504 | 0.783 | 0.614 | 0.453 |
| Decision Tree | 0.740 | 0.830 | 0.507 | 0.789 | 0.617 | 0.459 |
| KNN | 0.765 | 0.802 | 0.560 | 0.535 | 0.547 | 0.389 |
| Naive Bayes | 0.695 | 0.807 | 0.459 | 0.837 | 0.593 | 0.424 |
| Support Vector Machine | 0.737 | 0.839 | 0.503 | 0.786 | 0.614 | 0.453 |

The Streamlit app recalculates and displays each model's metrics on the selected labeled test CSV.

### Observations on performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Highest AUC (0.841), with strong recall (0.783) and MCC (0.453). It separates churn risk well while identifying many churners. |
| Decision Tree | Highest F1 (0.617) and MCC (0.459), with recall of 0.789. Its accuracy is 0.740; the depth limit constrains tree complexity. |
| KNN | Best accuracy (0.765) and precision (0.560), but lower recall (0.535) means it misses more actual churners than the recall-oriented models. |
| Naive Bayes | High recall (0.837), with accuracy 0.695 and AUC 0.807; this is a much more balanced baseline after treating total charges as numeric. |
| Support Vector Machine | The linear SVM is fast to score and has AUC 0.839, F1 0.614, and recall 0.786—close to logistic regression on this split. |
| **Overall, winner for your dataset?** | Decision Tree has the best F1 (0.617) and MCC (0.459); Logistic Regression has the best AUC (0.841). The best choice depends on whether balanced classification or ranking is the priority. |

## Project structure

```text
Telecom_Churn_Classification/
├── app.py
├── train_models.py
├── requirements.txt
├── README.md
├── test_data.csv
└── model/
    ├── metadata.json
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knearest_neighbors.joblib
    ├── naive_bayes.joblib
    └── support_vector_machine.joblib
```

## Run locally

1. Install the packages listed in `requirements.txt`.
2. Run `python train_models.py` once to build the five saved pipelines and the reproducible held-out test CSV.
3. Run `streamlit run app.py` and open the local URL printed in the terminal (normally `http://localhost:8501`).
4. Select a model, upload a labeled test CSV (or use the bundled test set), and inspect the metrics, confusion matrix, and classification report.

The app does not train or refit models. All preprocessing and classifier steps are persisted together in each model pipeline. The trainer reads the supplied sample CSV from its sibling `Sample Data/Classification/Telecome Customer Churn/` folder; the saved models and test CSV are included for running the app independently.

## Deploy to Streamlit Community Cloud

1. Create a GitHub repository and push the contents of this project folder to its root. Keep `app.py`, `requirements.txt`, `test_data.csv`, and the complete `model/` directory in the repository; do not upload `.venv`.
2. Open [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with the GitHub account that can access the repository.
3. Select **Create app**, choose the repository and branch, and set the main file path to `app.py`.
4. Select **Deploy** and wait for the build to finish. The app should load using the committed model artifacts; model training is not required in the cloud.
5. Open the resulting public app URL and verify model selection, metrics, CSV upload, confusion matrix, and classification report.
6. Add both the GitHub repository URL and deployed app URL to the submission PDF and the links below.

- **GitHub repository:** https://github.com/2026ak05272/ML_Assignment2
- **Live Streamlit app:** [https://telecomecustomerchurn.streamlit.app/](https://telecomecustomerchurn.streamlit.app/)

## Submission items still to complete

- Capture the required BITS Virtual Lab screenshot and include it in the final submission PDF.