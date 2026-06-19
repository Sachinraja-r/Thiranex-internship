# Executive EDA Report: E-commerce Customer Behavior & Churn

**Prepared by:** College Internship Project
**Dataset Size:** 1000 Rows | 13 Columns
**Objective:** Clean the raw purchase logs, inspect statistical properties, and identify key drivers of customer spending and business churn.

---

## 1. Dataset Overview & Schema
The dataset contains transaction records and demographic details for 1,000 customers. Below is the summary of variables:

| Column Name | Data Type | Non-Null Count | Missing Value % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Customer_ID` | Object (ID) | 1000 | 0.0% | Unique identifier for each customer |
| `Signup_Date` | Datetime | 1000 | 0.0% | Account registration date |
| `Age` | Float | 1000 | 0.0% (Imputed) | Customer age in years |
| `Gender` | Object | 1000 | 0.0% | Gender identification |
| `Annual_Income` | Integer | 1000 | 0.0% | Annual income of the customer (USD) |
| `Spending_Score` | Integer | 1000 | 0.0% | Score (1-100) indicating purchasing velocity |
| `Membership_Type`| Object | 1000 | 0.0% | Bronze, Silver, Gold, or Premium tier |
| `Preferred_Category`| Object | 1000 | 0.0% | Most shopped product category |
| `Total_Purchases`| Integer | 1000 | 0.0% | Total orders in the last 12 months |
| `Total_Spent` | Float | 1000 | 0.0% | Cumulative spending value (USD) |
| `Last_Active_Days`| Integer | 1000 | 0.0% | Days since last website interaction |
| `Satisfaction_Score`| Float | 1000 | 0.0% (Imputed) | Customer survey feedback score (1-5) |
| `Churn` | Integer | 1000 | 0.0% | Target Variable (1 = Churned, 0 = Retained) |

---

## 2. Data Quality & Cleaning
Real-world datasets contain anomalies. In this project, we performed the following cleaning procedures:

### A. Missing Value Imputation
* **`Age` (Missing 50 values):** Imputed using the **Median** (35.0 years). The median was selected because it is robust against extreme outliers.
* **`Satisfaction_Score` (Missing 40 values):** Imputed using the **Mode** (4.0). Since satisfaction score is an ordinal rating, using the most frequent rating preserves its discrete nature.
* *Visualization reference:* [Missing Values Analysis Plot](plots/01_missing_values.png)

### B. Anomaly & Outlier Correction
* **Age Boundaries:** Detected 3 records with anomalous ages (e.g. negative or > 100 years). These were replaced with the median value.
* **Financial Outliers (IQR Method):**
  * We identified outliers using the **1.5 * IQR rule** ($IQR = Q3 - Q1$).
  * **`Annual_Income`:** Upper boundary was $114,587.12. Found 25 outliers (high earners).
  * **`Total_Spent`:** Upper boundary was $1,333.35. Found 31 outliers (high spenders).
  * *Visualization reference:* [Financial Outliers Box Plot](plots/03_financial_outliers_boxplot.png)

---

## 3. Key Statistical Findings

### Customer Demographics
* The customer base ranges from 18 to 85 years old, with a mean age of **35.2 years**.
* **Preferred Shopping Categories:** **Clothing** is the most popular shopping category with **319** users, followed closely by **Electronics**.

### Financial and Spending Behavior
* Median annual customer income stands at **$53,411.00**.
* Total spending has a strong linear relationship with the number of purchases.
* A scatter plot segmenting Income vs Spent by Membership shows that Gold and Premium customers spend significantly more, grouping tightly at higher price ranges.
* *Visualization reference:* [Income vs Spent Scatter Plot](plots/07_income_vs_spent.png)

### Feature Correlations
Looking at the correlation heatmap, we observed these Pearson correlation coefficients:
* **`Total_Purchases` and `Total_Spent`:** **0.49** (Very strong positive correlation. Drive transaction frequency to increase revenue).
* **`Spending_Score` and `Total_Spent`:** **0.38** (Moderate-to-strong positive correlation).
* **`Satisfaction_Score` and `Churn`:** **-0.37** (Strong negative correlation. Lower satisfaction strongly correlates with churning).
* *Visualization reference:* [Correlation Heatmap](plots/08_correlation_heatmap.png)

---

## 4. Why Do Customers Churn? (Influencing Factors)
The overall churn rate in this cohort is **44.0%**.

### Factor 1: Customer Dissatisfaction
* Customers who rated their satisfaction as **1.0** had a churn rate of **81.2%**.
* In contrast, customers who rated their satisfaction as **5.0** had a churn rate of only **21.9%**.
* *Visualization reference:* [Churn by Satisfaction Score](plots/09_churn_by_satisfaction.png)

### Factor 2: Platform Inactivity
* Retained customers had an average inactivity of **74.8 days**.
* Churned customers had a significantly higher average inactivity of **109.2 days**.
* *Visualization reference:* [Churn by Inactivity Days Box Plot](plots/10_churn_by_activity.png)

---

## 5. Business Recommendations (For Internship Submission)
1. **Target Inactive Customers Early:** Implement an automated re-engagement campaign (via email or discount codes) when a customer exceeds **60 days of inactivity**, as inactivity is a leading indicator of churn.
2. **Prioritize Low-Satisfaction Alerts:** Customer service should immediately follow up with users who submit a satisfaction score of **1 or 2**. Resolving their complaints can mitigate the high churn risk (81.2%).
3. **Promote Membership Upgrades:** Since Gold and Premium members exhibit much higher total spending, creating marketing promotions to upgrade Silver members to Gold could boost transaction volume.
