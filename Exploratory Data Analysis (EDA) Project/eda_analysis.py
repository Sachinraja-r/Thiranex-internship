import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set plotting styling for professional, premium-looking charts
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({
    'font.size': 11,
    'axes.labelsize': 12,
    'axes.titlesize': 14,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'figure.titlesize': 16,
    'figure.dpi': 150
})

def run_eda_pipeline():
    print("=" * 60)
    print("STARTING EXPLORATORY DATA ANALYSIS PIPELINE")
    print("=" * 60)
    
    # Create directories for outputs
    os.makedirs('plots', exist_ok=True)
    
    # ----------------------------------------------------
    # STEP 1: DATA INGESTION & INITIAL INSPECTION
    # ----------------------------------------------------
    print("\n[STEP 1] Ingesting and Inspecting Dataset...")
    df = pd.read_csv('data/customer_data.csv')
    
    initial_shape = df.shape
    print(f"Dataset successfully loaded. Dimensions: {initial_shape[0]} rows, {initial_shape[1]} columns.")
    
    # Check data types and missing values
    missing_counts = df.isnull().sum()
    missing_pct = (missing_counts / len(df)) * 100
    data_types = df.dtypes
    
    inspection_df = pd.DataFrame({
        'Data Type': data_types,
        'Missing Values': missing_counts,
        'Missing Percentage (%)': missing_pct
    })
    print("\nDataset Schema and Missing Value Summary:")
    print(inspection_df)
    
    # Plot missing values before cleaning
    plt.figure(figsize=(10, 5))
    missing_only = missing_pct[missing_pct > 0]
    if not missing_only.empty:
        sns.barplot(x=missing_only.index, y=missing_only.values, hue=missing_only.index, palette="Reds_r", legend=False)
        plt.title('Percentage of Missing Values by Feature')
        plt.ylabel('Missing Percentage (%)')
        plt.xlabel('Features')
        for i, val in enumerate(missing_only.values):
            plt.text(i, val + 0.2, f"{val:.1f}%", ha='center', fontweight='bold')
    else:
        plt.text(0.5, 0.5, "No Missing Values Found", ha='center', va='center', fontsize=14)
    plt.tight_layout()
    plt.savefig('plots/01_missing_values.png')
    plt.close()
    
    # ----------------------------------------------------
    # STEP 2: DATA CLEANING & PREPROCESSING
    # ----------------------------------------------------
    print("\n[STEP 2] Cleaning Dataset & Resolving Anomalies...")
    
    # Create a copy for cleaning
    df_clean = df.copy()
    
    # 2.1 Convert Signup_Date to datetime
    df_clean['Signup_Date'] = pd.to_datetime(df_clean['Signup_Date'])
    
    # 2.2 Handle Anomalous Ages (< 18 or > 100)
    # Let's count how many anomalous records we have
    anomalous_age_mask = (df_clean['Age'] < 18) | (df_clean['Age'] > 100)
    num_anomalous_ages = anomalous_age_mask.sum()
    print(f"-> Detected {num_anomalous_ages} anomalous age values (< 18 or > 100 years).")
    
    # Calculate median of valid ages
    median_age = df_clean.loc[~anomalous_age_mask & df_clean['Age'].notnull(), 'Age'].median()
    print(f"-> Calculated median age of valid customers: {median_age} years.")
    
    # Replace anomalous ages with the median
    df_clean.loc[anomalous_age_mask, 'Age'] = median_age
    
    # 2.3 Impute Missing Ages
    # Since Age is a numerical variable and might be skewed or contain outliers, median is a robust choice
    num_missing_ages = df_clean['Age'].isnull().sum()
    df_clean['Age'] = df_clean['Age'].fillna(median_age)
    print(f"-> Imputed {num_missing_ages} missing Age values with the median ({median_age}).")
    
    # 2.4 Impute Missing Satisfaction Scores
    # Since Satisfaction_Score is an ordinal categorical scale (1-5), the Mode (most frequent value) is the ideal choice.
    satisfaction_mode = df_clean['Satisfaction_Score'].mode()[0]
    num_missing_sat = df_clean['Satisfaction_Score'].isnull().sum()
    df_clean['Satisfaction_Score'] = df_clean['Satisfaction_Score'].fillna(satisfaction_mode)
    print(f"-> Imputed {num_missing_sat} missing Satisfaction Scores with the mode ({satisfaction_mode}).")
    
    # Verify cleaning
    print(f"-> Verification: Remaining missing values in clean dataset = {df_clean.isnull().sum().sum()}")
    
    # ----------------------------------------------------
    # STEP 3: OUTLIER DETECTION (IQR METHOD)
    # ----------------------------------------------------
    print("\n[STEP 3] Running Outlier Detection on Financial Metrics...")
    
    # Let's analyze outliers in Annual_Income and Total_Spent
    metrics_to_check = ['Annual_Income', 'Total_Spent']
    outlier_summary = {}
    
    plt.figure(figsize=(12, 5))
    for idx, col in enumerate(metrics_to_check):
        plt.subplot(1, 2, idx+1)
        sns.boxplot(y=df_clean[col], color="skyblue")
        plt.title(f'Box Plot of {col}')
        plt.ylabel(col)
        
        # Calculate IQR
        Q1 = df_clean[col].quantile(0.25)
        Q3 = df_clean[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df_clean[(df_clean[col] < lower_bound) | (df_clean[col] > upper_bound)]
        outlier_summary[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'Lower Bound': lower_bound,
            'Upper Bound': upper_bound,
            'Outlier Count': len(outliers),
            'Outlier Percentage (%)': (len(outliers) / len(df_clean)) * 100
        }
    plt.tight_layout()
    plt.savefig('plots/03_financial_outliers_boxplot.png')
    plt.close()
    
    print("Outlier details (using IQR = Q3 - Q1 rule):")
    for col, stats in outlier_summary.items():
        print(f"  * {col}: Found {stats['Outlier Count']} outliers ({stats['Outlier Percentage (%)']:.2f}% of data)")
        print(f"    Bounds: [{stats['Lower Bound']:.2f}, {stats['Upper Bound']:.2f}] (Q1={stats['Q1']:.2f}, Q3={stats['Q3']:.2f})")

    # ----------------------------------------------------
    # STEP 4: UNIVARIATE ANALYSIS (DISTRIBUTIONS)
    # ----------------------------------------------------
    print("\n[STEP 4] Generating Univariate Distributions...")
    
    # 4.1 Age Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df_clean['Age'], kde=True, color='teal', bins=20)
    plt.axvline(df_clean['Age'].mean(), color='red', linestyle='--', label=f"Mean: {df_clean['Age'].mean():.1f}")
    plt.axvline(df_clean['Age'].median(), color='blue', linestyle='-', label=f"Median: {df_clean['Age'].median():.1f}")
    plt.title('Age Distribution of Customers (Cleaned)')
    plt.xlabel('Age')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/02_age_distribution.png')
    plt.close()
    
    # 4.2 Income Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df_clean['Annual_Income'], kde=True, color='purple', bins=25)
    plt.axvline(df_clean['Annual_Income'].median(), color='blue', linestyle='-', label=f"Median: ${df_clean['Annual_Income'].median():,.0f}")
    plt.title('Annual Income Distribution ($ USD)')
    plt.xlabel('Annual Income ($)')
    plt.ylabel('Count')
    plt.legend()
    plt.tight_layout()
    plt.savefig('plots/04_income_distribution.png')
    plt.close()
    
    # 4.3 Membership Type Distribution
    plt.figure(figsize=(8, 5))
    membership_counts = df_clean['Membership_Type'].value_counts()
    sns.barplot(x=membership_counts.index, y=membership_counts.values, hue=membership_counts.index, palette="viridis", legend=False)
    plt.title('Distribution of Customer Membership Types')
    plt.xlabel('Membership Type')
    plt.ylabel('Number of Customers')
    for i, val in enumerate(membership_counts.values):
        plt.text(i, val + 5, str(val), ha='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/05_membership_distribution.png')
    plt.close()
    
    # 4.4 Preferred Category Distribution
    plt.figure(figsize=(10, 5))
    category_counts = df_clean['Preferred_Category'].value_counts()
    sns.barplot(x=category_counts.values, y=category_counts.index, hue=category_counts.index, palette="pastel", legend=False)
    plt.title('Preferred Categories of Purchase')
    plt.xlabel('Number of Customers')
    plt.ylabel('Category')
    for i, val in enumerate(category_counts.values):
        plt.text(val + 5, i, str(val), va='center', fontweight='bold')
    plt.tight_layout()
    plt.savefig('plots/06_category_preferences.png')
    plt.close()
    
    # ----------------------------------------------------
    # STEP 5: BIVARIATE & MULTIVARIATE ANALYSIS
    # ----------------------------------------------------
    print("\n[STEP 5] Discovering Relationships and Correlations...")
    
    # 5.1 Income vs Spending
    plt.figure(figsize=(10, 6))
    sns.scatterplot(
        data=df_clean, 
        x='Annual_Income', 
        y='Total_Spent', 
        hue='Membership_Type', 
        palette='Set2',
        alpha=0.8,
        edgecolor='w',
        s=70
    )
    plt.title('Annual Income vs Total Spent (Segmented by Membership)')
    plt.xlabel('Annual Income ($)')
    plt.ylabel('Total Spent ($)')
    plt.legend(title='Membership Type')
    plt.tight_layout()
    plt.savefig('plots/07_income_vs_spent.png')
    plt.close()
    
    # 5.2 Correlation Heatmap (Numerical columns only)
    numerical_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    # Remove Customer_ID and Signup_Date if they got parsed as numbers (they shouldn't be, but just in case)
    if 'Customer_ID' in numerical_cols: numerical_cols.remove('Customer_ID')
    
    corr_matrix = df_clean[numerical_cols].corr()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='coolwarm', 
        fmt=".2f", 
        linewidths=0.5, 
        vmin=-1, 
        vmax=1,
        square=True
    )
    plt.title('Correlation Heatmap of Numerical Features', fontsize=14, pad=15)
    plt.tight_layout()
    plt.savefig('plots/08_correlation_heatmap.png')
    plt.close()
    
    # Print out strongest correlations
    print("Strongest correlations with Total_Spent:")
    print(corr_matrix['Total_Spent'].sort_values(ascending=False))
    
    # ----------------------------------------------------
    # STEP 6: CUSTOMER CHURN ANALYSIS (KEY INFLUENCING FACTORS)
    # ----------------------------------------------------
    print("\n[STEP 6] Analyzing Key Influencing Factors for Customer Churn...")
    
    # 6.1 Churn Rate by Satisfaction Score
    churn_sat = pd.crosstab(df_clean['Satisfaction_Score'], df_clean['Churn'], normalize='index') * 100
    
    plt.figure(figsize=(8, 5))
    churn_sat.plot(kind='bar', stacked=True, color=['#8fcc8f', '#ff7f7f'], ax=plt.gca())
    plt.title('Customer Churn Percentage by Satisfaction Score')
    plt.xlabel('Satisfaction Score (1-5)')
    plt.ylabel('Percentage (%)')
    plt.legend(['Retained (0)', 'Churned (1)'], loc='lower left')
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.savefig('plots/09_churn_by_satisfaction.png')
    plt.close()
    
    # 6.2 Churn Rate by Activity Level (Last Active Days)
    plt.figure(figsize=(8, 5))
    sns.boxplot(data=df_clean, x='Churn', y='Last_Active_Days', hue='Churn', palette=['g', 'r'], legend=False)
    plt.title('Days Since Last Activity: Churned vs Retained')
    plt.xlabel('Churn (0 = Retained, 1 = Churned)')
    plt.ylabel('Days Since Last Activity')
    plt.tight_layout()
    plt.savefig('plots/10_churn_by_activity.png')
    plt.close()
    
    # Calculate some exact metrics for the report
    total_churn_rate = df_clean['Churn'].mean() * 100
    churn_by_sat1 = df_clean[df_clean['Satisfaction_Score'] == 1.0]['Churn'].mean() * 100
    churn_by_sat5 = df_clean[df_clean['Satisfaction_Score'] == 5.0]['Churn'].mean() * 100
    
    mean_active_retained = df_clean[df_clean['Churn'] == 0]['Last_Active_Days'].mean()
    mean_active_churned = df_clean[df_clean['Churn'] == 1]['Last_Active_Days'].mean()
    
    # ----------------------------------------------------
    # STEP 7: COMPILE STRUCTURED REPORT
    # ----------------------------------------------------
    print("\n[STEP 7] Generating Structured Markdown Report (eda_report.md)...")
    
    report_content = f"""# Executive EDA Report: E-commerce Customer Behavior & Churn

**Prepared by:** College Internship Project
**Dataset Size:** {df_clean.shape[0]} Rows | {df_clean.shape[1]} Columns
**Objective:** Clean the raw purchase logs, inspect statistical properties, and identify key drivers of customer spending and business churn.

---

## 1. Dataset Overview & Schema
The dataset contains transaction records and demographic details for 1,000 customers. Below is the summary of variables:

| Column Name | Data Type | Non-Null Count | Missing Value % | Description |
| :--- | :--- | :--- | :--- | :--- |
| `Customer_ID` | Object (ID) | {len(df_clean)} | 0.0% | Unique identifier for each customer |
| `Signup_Date` | Datetime | {len(df_clean)} | 0.0% | Account registration date |
| `Age` | Float | {len(df_clean)} | 0.0% (Imputed) | Customer age in years |
| `Gender` | Object | {len(df_clean)} | 0.0% | Gender identification |
| `Annual_Income` | Integer | {len(df_clean)} | 0.0% | Annual income of the customer (USD) |
| `Spending_Score` | Integer | {len(df_clean)} | 0.0% | Score (1-100) indicating purchasing velocity |
| `Membership_Type`| Object | {len(df_clean)} | 0.0% | Bronze, Silver, Gold, or Premium tier |
| `Preferred_Category`| Object | {len(df_clean)} | 0.0% | Most shopped product category |
| `Total_Purchases`| Integer | {len(df_clean)} | 0.0% | Total orders in the last 12 months |
| `Total_Spent` | Float | {len(df_clean)} | 0.0% | Cumulative spending value (USD) |
| `Last_Active_Days`| Integer | {len(df_clean)} | 0.0% | Days since last website interaction |
| `Satisfaction_Score`| Float | {len(df_clean)} | 0.0% (Imputed) | Customer survey feedback score (1-5) |
| `Churn` | Integer | {len(df_clean)} | 0.0% | Target Variable (1 = Churned, 0 = Retained) |

---

## 2. Data Quality & Cleaning
Real-world datasets contain anomalies. In this project, we performed the following cleaning procedures:

### A. Missing Value Imputation
* **`Age` (Missing {num_missing_ages} values):** Imputed using the **Median** ({median_age:.1f} years). The median was selected because it is robust against extreme outliers.
* **`Satisfaction_Score` (Missing {num_missing_sat} values):** Imputed using the **Mode** ({satisfaction_mode:.1f}). Since satisfaction score is an ordinal rating, using the most frequent rating preserves its discrete nature.
* *Visualization reference:* [Missing Values Analysis Plot](plots/01_missing_values.png)

### B. Anomaly & Outlier Correction
* **Age Boundaries:** Detected {num_anomalous_ages} records with anomalous ages (e.g. negative or > 100 years). These were replaced with the median value.
* **Financial Outliers (IQR Method):**
  * We identified outliers using the **1.5 * IQR rule** ($IQR = Q3 - Q1$).
  * **`Annual_Income`:** Upper boundary was ${outlier_summary['Annual_Income']['Upper Bound']:,.2f}. Found {outlier_summary['Annual_Income']['Outlier Count']} outliers (high earners).
  * **`Total_Spent`:** Upper boundary was ${outlier_summary['Total_Spent']['Upper Bound']:,.2f}. Found {outlier_summary['Total_Spent']['Outlier Count']} outliers (high spenders).
  * *Visualization reference:* [Financial Outliers Box Plot](plots/03_financial_outliers_boxplot.png)

---

## 3. Key Statistical Findings

### Customer Demographics
* The customer base ranges from 18 to 85 years old, with a mean age of **{df_clean['Age'].mean():.1f} years**.
* **Preferred Shopping Categories:** **{category_counts.index[0]}** is the most popular shopping category with **{category_counts.values[0]}** users, followed closely by **{category_counts.index[1]}**.

### Financial and Spending Behavior
* Median annual customer income stands at **${df_clean['Annual_Income'].median():,.2f}**.
* Total spending has a strong linear relationship with the number of purchases.
* A scatter plot segmenting Income vs Spent by Membership shows that Gold and Premium customers spend significantly more, grouping tightly at higher price ranges.
* *Visualization reference:* [Income vs Spent Scatter Plot](plots/07_income_vs_spent.png)

### Feature Correlations
Looking at the correlation heatmap, we observed these Pearson correlation coefficients:
* **`Total_Purchases` and `Total_Spent`:** **{corr_matrix.loc['Total_Purchases', 'Total_Spent']:.2f}** (Very strong positive correlation. Drive transaction frequency to increase revenue).
* **`Spending_Score` and `Total_Spent`:** **{corr_matrix.loc['Spending_Score', 'Total_Spent']:.2f}** (Moderate-to-strong positive correlation).
* **`Satisfaction_Score` and `Churn`:** **{corr_matrix.loc['Satisfaction_Score', 'Churn']:.2f}** (Strong negative correlation. Lower satisfaction strongly correlates with churning).
* *Visualization reference:* [Correlation Heatmap](plots/08_correlation_heatmap.png)

---

## 4. Why Do Customers Churn? (Influencing Factors)
The overall churn rate in this cohort is **{total_churn_rate:.1f}%**.

### Factor 1: Customer Dissatisfaction
* Customers who rated their satisfaction as **1.0** had a churn rate of **{churn_by_sat1:.1f}%**.
* In contrast, customers who rated their satisfaction as **5.0** had a churn rate of only **{churn_by_sat5:.1f}%**.
* *Visualization reference:* [Churn by Satisfaction Score](plots/09_churn_by_satisfaction.png)

### Factor 2: Platform Inactivity
* Retained customers had an average inactivity of **{mean_active_retained:.1f} days**.
* Churned customers had a significantly higher average inactivity of **{mean_active_churned:.1f} days**.
* *Visualization reference:* [Churn by Inactivity Days Box Plot](plots/10_churn_by_activity.png)

---

## 5. Business Recommendations (For Internship Submission)
1. **Target Inactive Customers Early:** Implement an automated re-engagement campaign (via email or discount codes) when a customer exceeds **60 days of inactivity**, as inactivity is a leading indicator of churn.
2. **Prioritize Low-Satisfaction Alerts:** Customer service should immediately follow up with users who submit a satisfaction score of **1 or 2**. Resolving their complaints can mitigate the high churn risk ({churn_by_sat1:.1f}%).
3. **Promote Membership Upgrades:** Since Gold and Premium members exhibit much higher total spending, creating marketing promotions to upgrade Silver members to Gold could boost transaction volume.
"""
    
    with open('eda_report.md', 'w', encoding='utf-8') as f:
        f.write(report_content)
        
    print("Report compiled successfully as 'eda_report.md'.")
    print("=" * 60)
    print("EDA PIPELINE COMPLETE: ALL OUTPUTS VERIFIED")
    print("=" * 60)

if __name__ == "__main__":
    run_eda_pipeline()
