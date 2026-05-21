# Employee-Attrition

A supervised machine learning project that predicts whether an employee is likely to leave an organisation, using classification models trained on HR data. Built as part of an applied data science assignment for ICT Academy Kerala.

---

## 📌 Problem Statement

Employee attrition is a major operational challenge for IT services companies. High resignation rates lead to increased recruitment costs, loss of institutional knowledge, project delays, and reduced morale. The HR department currently relies on manual assumptions to identify at-risk employees — an approach that is both inefficient and reactive.

This project takes a data-driven approach: train machine learning models on historical employee data to **proactively identify employees at risk of leaving**, giving HR teams enough lead time to intervene.

---

## 📂 Dataset

| Property | Detail |
|---|---|
| Source | [Employee Attrition Prediction Dataset – Kaggle](https://www.kaggle.com/datasets/ziya07/employee-attrition-prediction-dataset) |
| Records | 1,000 employees |
| Features | 25 input features + 1 target |
| Target | `Attrition` → Yes (left) / No (stayed) |
| Class Balance | 81.1% stayed / 18.9% left ⚠️ Imbalanced |

### Key Features

| Category | Features |
|---|---|
| Demographics | Age, Gender, Marital Status, Distance From Home |
| Compensation | Monthly Income, Hourly Rate, Job Level |
| Work Factors | Overtime, Average Hours Worked Per Week, Project Count |
| Satisfaction | Job Satisfaction, Work-Life Balance, Work Environment Satisfaction, Relationship with Manager |
| Career | Years at Company, Years in Current Role, Years Since Last Promotion, Number of Companies Worked |
| Performance | Performance Rating, Training Hours Last Year, Job Involvement, Absenteeism |

---


## 📁 Repository Structure

```
employee-attrition-prediction/
│
├── Employee_Attrition_Prediction.ipynb   # Main notebook (all sections)
├── employee_attrition_dataset.csv        # Dataset
├── README.md
```

---

## 🔍 Project Workflow

### 1. Business Understanding
Defined the attrition problem in business terms — cost of turnover, impact on IT project delivery, and the value of early intervention. Identified key stakeholders as HR managers and department heads.

### 2. Data Understanding
- Dataset shape: 1,000 rows × 26 columns
- No missing values found
- No duplicate records
- Mix of numerical (continuous + ordinal) and categorical features
- Job Satisfaction rated 1–5; Work-Life Balance rated 1–4

### 3. Exploratory Data Analysis

Key findings from EDA :

| Feature | What the Data Shows |
|---|---|
| **Target (Attrition)** | 811 stayed (81.1%) vs 189 left (18.9%) — heavily imbalanced |
| **Department** | IT (21.3%) and HR (20.9%) have highest attrition; spread is only 4.2pp across all departments |
| **Overtime** | No overtime: 19.5% attrition, Overtime: 18.2% — difference of just 1.3pp, not a meaningful driver |
| **Monthly Income** | Employees who left had a *higher* median income (11,876) than those who stayed (11,226) |
| **Job Satisfaction** | Non-linear pattern — highest attrition at score 5 (20.9%), lowest at score 3 (16.8%) |
| **Work-Life Balance** | Counterintuitive — best WLB score (4) has highest attrition (21.5%); worst (1) has lowest (16.9%) |
| **Age** | Mean age difference of only 0.9 years between groups — not a predictor |
| **Absenteeism** | Mean difference of only 0.67 days between groups — very weak signal |
| **Correlations** | Maximum correlation with attrition is 0.056 (Training Hours) — all features are weak predictors |



### 4. Data Preprocessing

- Dropped `Employee_ID` — non-predictive identifier
- Encoded target: `Yes → 1`, `No → 0`
- Encoded `Overtime`: `Yes → 1`, `No → 0`
- One-hot encoded: `Gender`, `Marital_Status`, `Department`, `Job_Role`
- Applied `StandardScaler` for feature normalisation
- Stratified 80/20 train-test split to preserve class ratio (162 stayed / 38 left in test set)

### 5. Models Built

| Model | Configuration |
|---|---|
| Logistic Regression | max_iter=1000, random_state=42 |
| Decision Tree | max_depth=6, random_state=42 |
| Random Forest | n_estimators=200, max_depth=10, random_state=42 |

### 6. Model Evaluation

> ⚠️ **Class Imbalance Effect:** Logistic Regression and Random Forest both predicted everyone as "Stayed" — 81% accuracy but 0% recall on leavers. The Decision Tree was the only model that flagged any attrition at all.

| Model | Accuracy | Precision (Left) | Recall (Left) | F1 (Left) |
|---|---|---|---|---|
| Logistic Regression | 0.81 | 0.00 | 0.00 | 0.00 |
| **Decision Tree** | **0.72** | **0.09** | **0.05** | **0.07** |
| Random Forest | 0.81 | 0.00 | 0.00 | 0.00 |

**Best model: Decision Tree** — the only model that identified any leavers (2 out of 38 caught).

**Why Recall is the most critical metric here:**
- False negative = missed leaver → lost retention opportunity → costly
- False positive = wrongly flagged employee → brief HR conversation → cheap
- Therefore catching leavers (Recall) matters far more than overall accuracy

### 7. Feature Importance (Random Forest)

Top features by importance score — note all scores are close, confirming no dominant predictor:

| Rank | Feature | Importance |
|---|---|---|
| 1 | Monthly Income | 0.0798 |
| 2 | Hourly Rate | 0.0704 |
| 3 | Training Hours Last Year | 0.0697 |
| 4 | Distance From Home | 0.0644 |
| 5 | Absenteeism | 0.0622 |

---

## 💡 Business Insights & Recommendations

> These recommendations are grounded in what this dataset actually shows — not generic assumptions.

| What the Data Shows | Recommended HR Action |
|---|---|
| IT (21.3%) and HR (20.9%) have highest attrition | Focus retention efforts on these departments first; run role-specific surveys |
| Only 4.2pp spread across departments | Attrition is an organisation-wide issue — use individual risk scores, not department-level assumptions |
| Overtime makes no meaningful difference (1.3pp gap) | Investigate qualitative factors instead — manager relationships, career growth, role clarity |
| Employees who left earned more than those who stayed | Salary is not the retention lever here — focus on non-financial factors: autonomy, recognition, growth |
| Job satisfaction has no clean relationship with attrition | Combine satisfaction surveys with exit interview data for a fuller picture |
| Absenteeism difference is only 0.67 days | Monitor trends over time — sudden spikes matter more than absolute levels |

---


## 🚧 Limitations & Future Work

- **Class imbalance** is the primary limitation — apply SMOTE or `class_weight='balanced'` to improve minority class Recall
- **Weak features** — no single feature has a correlation above 0.056 with attrition; collecting richer data (exit interview themes, manager ratings, engagement scores) would help significantly
- **Small attrition sample** — only 189 attrition cases; more data would improve model reliability
- **Hyperparameter tuning** via GridSearchCV would squeeze more performance from existing models
- **XGBoost / LightGBM** could be tested as stronger alternatives to Random Forest
- **SHAP values** would provide per-employee explanations of attrition risk — more actionable for HR
- A **real-time scoring pipeline** could embed the model into an HR dashboard for monthly risk updates

---


## 📄 License

This project is for educational purposes as part of an ICT Academy Kerala assignment.
