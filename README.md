# Customer Churn Prediction & Retention Strategy

A full-stack churn analysis project built on 10,000 bank customer records. This covers both the machine learning side (Python, 5 models benchmarked) and the business analytics side (SQL, 9 business questions answered). The goal was to build something a retention team could actually use, not just a notebook that hits 86% accuracy and stops there.

---

## What's in this repo

| File | What it does |
|---|---|
| `churn_prediction_model.ipynb` | End-to-end ML pipeline: EDA, model benchmarking, feature importance, business recommendations |
| `churn_prediction.sql` | 9 SQL queries covering cohort analysis, risk scoring, high-value customer identification, and business impact sizing |
| `Churn_Modelling.csv` | Source dataset (10,000 customers, 14 features) |
| `requirements.txt` | Python dependencies |
| `visuals/` | Charts exported from the notebook |

---

## The business problem

The bank is losing roughly 20% of customers per year. At $243 average acquisition cost per customer, that is nearly $500K in replacement cost on this dataset alone, before accounting for lost deposits or cross-sell revenue.

The project answers two questions a retention team actually needs:
1. Which customers are most likely to leave in the next cycle?
2. Which ones should we prioritize given balance at risk?

---

## Model results

Five classifiers benchmarked on the same 80/20 train/test split:

| Model | Accuracy | Recall (Churn) | False Negatives |
|---|---|---|---|
| Gradient Boosting | 86.75% | 0.49 | 201 |
| Random Forest | 86.65% | 0.46 | 211 |
| KNN | 83.00% | 0.37 | 247 |
| Logistic Regression | 81.10% | 0.20 | 314 |
| SVM | 80.35% | 0.00 | 393 |

Accuracy alone is misleading here given the 4:1 class imbalance. Recall on the churn class is what actually matters. SVM scores 80% by predicting nobody churns, which is useless. Random Forest is the selected model based on the best balance of precision and recall.

---

## SQL analysis highlights

The companion SQL file goes beyond model output into business questions:

- **Q3/Q4:** Rules-based composite risk score that segments customers into Critical/High/Medium/Low bands. The Critical band churns at 45.5% vs a 20.4% baseline, a 7x difference that holds up without any ML.
- **Q5:** 785 high-value inactive customers sitting above $140K in balance, 40% of whom have already churned.
- **Q9:** Germany accounts for 53% of total balance at risk ($97.9M) despite being 25% of the customer base.

---

## Key findings

1. Age (46-60 band) and inactivity are the strongest churn signals, not credit score
2. Germany churns at 32.4% vs roughly 16% in France and Spain, a market issue not a demographic one
3. Single-product customers churn at 28%. Two-product customers at 8%. Cross-sell is a retention strategy.
4. Feature engineering (26 features) did not improve on the baseline (11 features), which means the original behavioral signals already capture most of the available information

---

## How to run

```bash
pip install -r requirements.txt
jupyter notebook churn_prediction_model.ipynb
```

Place `Churn_Modelling.csv` in the same directory as the notebook before running.

---

## Stack

Python, scikit-learn, pandas, matplotlib, MySQL
