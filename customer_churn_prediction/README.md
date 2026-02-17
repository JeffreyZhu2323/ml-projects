<img width="339" height="97" alt="image" src="https://github.com/user-attachments/assets/279cb6ed-d1ae-435e-90ec-010ea970a1cb" /># Customer Churn Prediction + Retention Targeting (Telco)

Predict customer churn and turn model scores into actionable retention targeting strategies. This project builds an end-to-end churn pipeline (cleaning → encoding → model training → evaluation), then translates model outputs into **operating points** (e.g., target top 10% highest-risk customers) and **business recommendations** based on the strongest churn drivers.

## TL;DR Results (Test Set)

**Model:** Logistic Regression (`class_weight="balanced"`, `max_iter=2000`)  
**Test ROC-AUC:** 0.8498  
**Test PR-AUC:** 0.6461 (test churn rate is ~26.5%, so this is far above baseline)

## Practical Operating Points (Threshold Policies)

Your `results/baseline_metrics` file stores multiple threshold policies so the model can be used in real operations (“who do we contact?”).

| Policy | Threshold | % Targeted | Precision | Recall | F1 | Confusion Matrix (TN, FP / FN, TP) |
|---|---:|---:|---:|---:|---:|---|
| Default | 0.50 | 40.7% | 0.513 | 0.786 | 0.621 | `[[756, 279], [80, 294]]` |
| Capacity-based (Top 10%) | 0.8474 | 10.0% | 0.752 | 0.283 | 0.412 | `[[1000, 35], [268, 106]]` |
| **Best F1 (picked on VAL, evaluated on TEST)** | **0.7048** | **26.3%** | **0.620** | **0.615** | **0.617** | `[[894, 141], [144, 230]]` |
| Business EV example (illustrative) | 0.02 | 91.1% | 0.292 | 1.000 | 0.451 | `[[126, 909], [0, 374]]` |

> Note: the “Business EV example” uses illustrative assumptions (contact_cost/value_saved). Real retention economics (cost, CLV, uplift) vary by company—this row is included to demonstrate expected-value thresholding.

---

## Dataset & Target

- **Dataset:** Telco customer churn  
- **Target:** `Churn Value` (0/1)

This pipeline produces **23 engineered features** across customer attributes, subscription/services, billing, and contract/payment.

---

## Approach

### Preprocessing

- Convert numeric columns (`Monthly Charges`, `Total Charges`) to numeric
- Handle `Total Charges` missing values when `Tenure Months == 0` by setting to 0 (new customers)
- Encode binary features (Yes/No, Male/Female) to 0/1
- One-hot encode multiclass features with `drop_first=True`:
  - `Internet Service`, `Contract`, `Payment Method`
- Standardize continuous features using **train mean/std only** (prevents leakage)
- Train/val/test split with **stratification** to preserve churn rate

### Model

- **Logistic Regression**
  - `class_weight="balanced"`
  - `max_iter=2000`
---

## Interpretation: Key Churn Drivers (Logistic Regression Coefficients)

Positive coefficients increase churn risk; negative coefficients decrease churn risk. Full coefficient list is saved at `results/variable_affect_churn.csv`.

### Strongest churn-increasing signals

- `Internet Service_Fiber optic` (+0.628)
- `Total Charges` (+0.478)
- `Payment Method_Electronic check` (+0.340)
- `Partner` (+0.269)
- `Multiple Lines` (+0.265)
- `Paperless Billing` (+0.253)
- `Streaming TV` (+0.201)
- `Streaming Movies` (+0.171)
- `Monthly Charges` (+0.117)

### Strongest churn-decreasing signals

- `Dependents` (−1.676)
- `Contract_Two year` (−1.536)
- `Tenure Months` (−1.173)
- `Contract_One year` (−0.811)
- `Phone Service` (−0.642)
- `Internet Service_No` (−0.602)
- `Tech Support` (−0.388)
- `Online Security` (−0.375)
- `Online Backup` (−0.151)
- `Device Protection` (−0.102)

**Takeaway:** churn risk is strongly associated with short tenure and short contracts, and higher churn among customers on fiber optic and electronic check. Support/security add-ons correlate with lower churn.

---

## Retention Recommendations (Actionable)

### 1) Prioritize month-to-month / short-tenure customers for proactive retention
**Why:** `Tenure Months` and long contracts are the strongest churn reducers.  
**Action:** offer contract upgrades (12–24 months) with limited-time incentives + an onboarding retention flow for the first 1–3 months.  
**Targeting:** high churn score + short tenure (or top-k highest risk).

### 2) Fiber optic customers: reduce service friction + bundle support
**Why:** fiber optic is the strongest positive churn signal.  
**Action:** provide a free Tech Support trial and targeted setup/troubleshooting content early in the customer lifecycle.  
**Targeting:** `Internet Service_Fiber optic = 1` + high churn score.

### 3) Encourage switching away from electronic check (promote autopay)
**Why:** electronic check payment is a strong churn-risk indicator.  
**Action:** incentivize autopay (credit card/bank transfer) with small discounts; reduce payment friction with reminders and clearer billing UX.  
**Targeting:** `Payment Method_Electronic check = 1` + medium/high churn score.

### 4) Bundle security/support add-ons for at-risk customers
**Why:** `Tech Support` and `Online Security` are associated with lower churn.  
**Action:** offer a limited-time bundle (Tech Support + Online Security) with guided setup to increase adoption.  
**Targeting:** high churn score AND currently lacking these add-ons.

### 5) Capacity-based targeting policy (simple + realistic)
If a retention team can contact a limited number of customers, targeting the **top 10% highest-risk** yields:
- Precision ≈ 0.75
- Recall ≈ 0.28
---

## Project Structure

```text
customer_churn_prediction/
  data/                      # place dataset here (ignored by git)
  reports/
    figures/
      pr_curve_test.png
      roc_curve_test.png
  results/
    baseline_metrics         # saved metrics (ROC/PR + threshold policies incl. best-F1)
    variable_affect_churn.csv
  src/
    __init__.py
    config.py
    data_loading.py          # preprocessing + train/val/test split
    training.py              # train + evaluate + write results/ + save plots
  README.md
  train.py                   # (optional) convenience entry point
  eval.py                    # (optional) convenience entry point
  .gitignore

