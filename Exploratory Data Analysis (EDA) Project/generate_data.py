import os
import numpy as np
import pandas as pd

def generate_customer_dataset(num_records=1000, seed=42):
    np.random.seed(seed)
    
    # 1. Customer ID
    customer_ids = [f"CUST{i:04d}" for i in range(1001, 1001 + num_records)]
    
    # 2. Signup Date (Datetime)
    start_date = pd.to_datetime('2024-01-01')
    end_date = pd.to_datetime('2025-12-31')
    days_range = (end_date - start_date).days
    random_days = np.random.randint(0, days_range, num_records)
    signup_dates = start_date + pd.to_timedelta(random_days, unit='D')
    # format signup_dates as YYYY-MM-DD
    signup_dates = signup_dates.strftime('%Y-%m-%d')
    
    # 3. Age (Normally distributed around 35, std dev 12)
    # Inject some missing values (5%) and explicit anomalies (outliers)
    age = np.random.normal(35, 12, num_records).astype(int)
    age = np.clip(age, 18, 85) # Normal realistic range
    
    # Introduce outliers
    age[15] = -5     # Negative age
    age[120] = 115   # Unrealistically high age
    age[240] = 125   # Unrealistically high age
    
    # Convert to float to support NaNs
    age_float = age.astype(float)
    # Inject missing values (NaNs)
    missing_age_idx = np.random.choice(num_records, size=int(0.05 * num_records), replace=False)
    age_float[missing_age_idx] = np.nan
    
    # 4. Gender (Categorical)
    genders = np.random.choice(['Male', 'Female', 'Non-Binary', 'Prefer not to say'], 
                               size=num_records, p=[0.47, 0.47, 0.04, 0.02])
    
    # 5. Annual Income (Log-normal to represent typical salary distribution)
    # Scale it to look like real USD ($20,000 to $180,000)
    income_base = np.random.lognormal(mean=11.0, sigma=0.4, size=num_records)
    annual_income = (income_base * 0.9).astype(int)
    # Ensure minimum income
    annual_income = np.clip(annual_income, 15000, 220000)
    
    # Inject high-income outliers
    annual_income[50] = 450000
    annual_income[300] = 480000
    annual_income[650] = 520000
    
    # 6. Spending Score (1 to 100)
    spending_score = np.random.randint(1, 101, num_records)
    
    # 7. Membership Type
    # High income and spending score tend to have Gold/Premium memberships
    memberships = []
    for inc, score in zip(annual_income, spending_score):
        score_val = inc / 2500 + score
        if score_val > 110:
            memberships.append(np.random.choice(['Gold', 'Premium'], p=[0.4, 0.6]))
        elif score_val > 60:
            memberships.append(np.random.choice(['Silver', 'Gold'], p=[0.6, 0.4]))
        else:
            memberships.append(np.random.choice(['Bronze', 'Silver'], p=[0.7, 0.3]))
            
    # 8. Preferred Category
    categories = np.random.choice(['Electronics', 'Clothing', 'Home & Kitchen', 'Beauty', 'Sports & Outdoors'], 
                                  size=num_records, p=[0.25, 0.30, 0.20, 0.15, 0.10])
    
    # 9. Total Purchases (related to spending score)
    total_purchases = np.random.poisson(lam=(spending_score / 8) + 2)
    total_purchases = np.clip(total_purchases, 1, 45)
    
    # 10. Total Spent (related to purchases and income)
    # Total spent = purchases * avg item value (correlated with income) + noise
    avg_item_value = (annual_income / 1200) + np.random.normal(15, 5, num_records)
    avg_item_value = np.clip(avg_item_value, 5, 200)
    total_spent = (total_purchases * avg_item_value).round(2)
    
    # Inject total spent outliers
    total_spent[75] = 9500.00
    total_spent[512] = 11200.00
    
    # 11. Last Active Days (0 to 180 days)
    last_active_days = np.random.randint(0, 181, num_records)
    
    # 12. Satisfaction Score (1 to 5)
    satisfaction_score = np.random.choice([1, 2, 3, 4, 5], size=num_records, p=[0.1, 0.15, 0.25, 0.35, 0.15]).astype(float)
    # Inject missing values (4%)
    missing_satisfaction_idx = np.random.choice(num_records, size=int(0.04 * num_records), replace=False)
    satisfaction_score[missing_satisfaction_idx] = np.nan
    
    # 13. Churn (Binary Target Variable, correlated with satisfaction and active days)
    churn = []
    for score, active_days, spec_score in zip(satisfaction_score, last_active_days, spending_score):
        # Calculate churn probability
        prob = 0.1 # Base probability
        
        # Low satisfaction increases churn
        if pd.isna(score):
            prob += 0.15
        elif score == 1:
            prob += 0.55
        elif score == 2:
            prob += 0.35
        elif score == 3:
            prob += 0.10
        elif score == 5:
            prob -= 0.08
            
        # Inactivity increases churn
        if active_days > 120:
            prob += 0.40
        elif active_days > 60:
            prob += 0.15
            
        # Low spending score/engagement increases churn
        if spec_score < 25:
            prob += 0.15
            
        # Clamp probability
        prob = np.clip(prob, 0.02, 0.98)
        churn.append(np.random.choice([0, 1], p=[1 - prob, prob]))
        
    # Build DataFrame
    df = pd.DataFrame({
        'Customer_ID': customer_ids,
        'Signup_Date': signup_dates,
        'Age': age_float,
        'Gender': genders,
        'Annual_Income': annual_income,
        'Spending_Score': spending_score,
        'Membership_Type': memberships,
        'Preferred_Category': categories,
        'Total_Purchases': total_purchases,
        'Total_Spent': total_spent,
        'Last_Active_Days': last_active_days,
        'Satisfaction_Score': satisfaction_score,
        'Churn': churn
    })
    
    return df

if __name__ == "__main__":
    print("Generating synthetic E-commerce Customer dataset...")
    df = generate_customer_dataset(1000)
    
    # Create data directory if not exists
    os.makedirs('data', exist_ok=True)
    
    # Save to CSV
    output_path = 'data/customer_data.csv'
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and saved to '{output_path}'")
    print(f"Shape of the dataset: {df.shape}")
    print("\nFirst 5 rows:")
    print(df.head())
    print("\nMissing values summary:")
    print(df.isnull().sum())
