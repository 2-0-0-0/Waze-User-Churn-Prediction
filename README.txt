🚗 Waze User Churn Prediction — Machine Learning for Retention Strategy
This project leverages supervised machine learning models to predict user churn for the Waze navigation app. By analyzing behavioral and engagement data, it identifies users likely to churn and helps Waze design proactive retention mechanisms to keep users engaged.

📌 Project Overview
Goal: Predict whether a Waze user will churn based on their app usage patterns and behavioral features.

Techniques Used: XGBoost, Random Forest, Ensemble Learning, SMOTE, SHAP (Explainability)

Outcome: Achieved 73% accuracy with enhanced recall for churned users through ensemble modeling.


📊 Dataset
14,999 user records with 13 features

Collected user behavior such as:

Driving frequency and duration

App engagement (sessions, activity days)

Device type (Android/iPhone)

Navigation to favorite locations

Target label: churned or retained

> Note: Data cleaning included handling 700 missing labels and one-hot encoding device types.

🧰 Tools & Libraries
Python (Pandas, NumPy, Scikit-learn)

XGBoost

Imbalanced-learn (SMOTE)

SHAP (Explainability)

Matplotlib, Seaborn (Visualization)


🧹 Data Preprocessing
Missing Data Handling: Mode imputation for target variable

Categorical Encoding: One-hot for device

Scaling: Standardized numerical features

SMOTE: To balance class distribution

Train/Test Split: 80/20 stratified sampling

📈 Exploratory Data Analysis (EDA)
Found that churned users often:

Had fewer activity days

Logged fewer drives and sessions

Navigated to favorites less frequently

Visualized using histograms, boxplots, and heatmaps

🧠 Modeling
Algorithms:
XGBoost (optimized with Grid Search + scale_pos_weight)

Random Forest (tuned with class_weight and hyperparameters)

Voting Classifier (Soft voting ensemble for stability)

Hyperparameter Tuning:
GridSearchCV on parameters such as:

n_estimators, max_depth, learning_rate, min_samples_split

📊 Evaluation Metrics
Model	Accuracy	Churn Recall	Churn Precision	F1-score
XGBoost	75.1%	39%	32%	35%
Random Forest	72.2%	52%	32%	40%
Ensemble	73.1%	46%	32%	38%
> Chosen Model: Ensemble classifier for balanced performance

🔍 Feature Importance
Analyzed using SHAP values. Key predictors of churn:

activity_days

duration_minutes_drives

driven_km_drives

sessions

drives

Visualized SHAP summary plots to guide feature selection and future app optimizations.