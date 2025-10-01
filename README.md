# Accredian_Internship Task
This project focuses on detecting fraudulent financial transactions using the PaySim synthetic dataset.
The goal is to build a robust machine learning pipeline to identify fraud, understand key predictive factors, and propose strategies to prevent fraudulent activities in financial systems.

# Tools & Libraries
**Python** 3.x
**Data Manipulation:** pandas, numpy
**Visualization:** matplotlib, seaborn
**Machine Learning:** scikit-learn, xgboost
**Imbalanced Data Handling:** imbalanced-learn (SMOTE)
**Jupyter Notebook** for step-by-step workflow and reproducibility

# Methodology
**1. Data Cleaning**
- Handle missing values (impute or drop)
- Detect and remove outliers using IQR method
- Check multicollinearity via Variance Inflation Factor (VIF) and drop highly correlated features if needed

**2. Exploratory Data Analysis (EDA)**
- Visualize distributions of transaction types and amounts
- Examine fraud ratio and patterns
- Identify anomalies and potential data issues

**3. Feature Engineering & Selection**
- One-hot encode categorical variables (transaction type)
- Scale numerical features
- Use correlation matrix, domain knowledge, and model feature importance to select key predictors

**4. Model Building**
- XGBoost Classifier selected for:
- Handling imbalanced datasets effectively
- Providing feature importance
- Train/test split: 80/20
- Oversample minority class using SMOTE
- Use cross-validation for robust performance estimates

**5. Model Evaluation**
- Metrics: AUC-ROC, Precision-Recall curve, F1-score, confusion matrix
- Focus on recall to minimize missed fraud cases
- Compare baseline vs. oversampled performance

**6. Feature Interpretation**
- Analyze feature importance
- Validate if predictive patterns make logical sense:
    - Fraudulent transactions often involve transfers, cash outs, or zeroing out balances

# How to Run the Notebook
1. Clone or download the repository.
2. Install required packages:
   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost shap
3. Place paysim.csv in the working directory.
4. Open Fraud_Detection_PaySim.ipynb in Jupyter Notebook.
5. Run all cells sequentially to reproduce the analysis.

# Results
- The model achieved high recall on fraud cases after SMOTE oversampling.
- Key predictive features: transaction type (TRANSFER/CASH_OUT),zero newbalanceOrig.
- Recommendations were formulated based on feature analysis and domain knowledge.

# Notes
- Dataset is large (~6.36M rows). For quick testing, sample a subset (e.g., 100k rows) to speed up experimentation.

# Author
- **Name:** Deepti Mahajan
- **Date:** 2025-10-01
- Internship-level project suitable for ML/data science portfolios.
