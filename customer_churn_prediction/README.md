# Customer Churn Prediction + Retention Targeting (Telco)

Predict customer churn and turn model scores into actionable retention targeting strategies. This project builds an end-to-end churn pipeline (cleaning → encoding → model training → evaluation), then translates model outputs into **operating points** (e.g., target top 10% highest-risk customers) and **business recommendations** based on the strongest churn drivers.

## TL;DR Results (Test Set)

**Model:** Logistic Regression (`class_weight="balanced"`)  
**Test ROC-AUC:** 0.850  
**Test PR-AUC:** 0.646 (churn is ~26–28%, so this is far above baseline)

### Practical operating points (examples)

I evaluate multiple decision thresholds because deployment requires a clear “who do we contact?” rule.

- **Threshold = 0.50**
  - Targeted: 40.7%
  - Precision: 0.513, Recall: 0.786, F1: 0.621
  - Confusion matrix: `[[756, 279], [80, 294]]`

- **Target top 10% highest-risk**
  - Targeted: 10.0%
  - Precision: 0.752, Recall: 0.283, F1: 0.412
  - Confusion matrix: `[[1000, 35], [268, 106]]`

- **Business EV example (illustrative assumptions)**
  - Threshold = 0.02 (contact_cost/value_saved = 1/50)
  - Targeted: 91.1%
  - Precision: 0.292, Recall: 1.000, F1: 0.451
  - Confusion matrix: `[[126, 909], [0, 374]]`
  - Note: this is included to demonstrate expected-value thresholding; real cost/CLV/uplift assumptions vary by company.

- **Best-F1 threshold (chosen on validation set)**
  - Stored in `reports/metrics_*.json` and evaluated on test using the same threshold.

---

## Dataset & Target

- **Dataset:** Telco customer churn  
- **Target:** `Churn Value` (0/1)

I use **23 engineered features** across customer attributes, subscription/services, billing, and contract/payment.

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

Logistic Regression is a strong churn baseline: fast, stable, and directly interpretable via coefficients.

---

## Interpretation: Key Churn Drivers (Logistic Regression Coefficients)

Positive coefficients increase churn risk; negative coefficients decrease churn risk.

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

**Why:** Fiber optic is the strongest positive churn signal.  
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

This is often more operationally realistic than a fixed threshold like 0.50.

---

## Project Structure

```text
customer_churn_prediction/
  src/
    data_loading.py
    training.py
  reports/
    metrics_*.json
    figures/
      roc_curve_test.png
      pr_curve_test.png
  models/
    (optional) saved_model.joblib
