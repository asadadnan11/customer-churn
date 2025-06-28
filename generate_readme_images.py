#!/usr/bin/env python3
"""
Generate key visualizations for README.md from the telecom churn prediction notebook
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style and random seed for reproducibility
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print("Generating visualizations for README...")

def generate_telecom_data(n_customers=10000):
    """Generate synthetic telecom customer data"""
    print(f"Generating synthetic data for {n_customers:,} customers...")
    
    # Initialize lists to store generated data
    data = {
        'customer_id': [f'CUST_{i+1:06d}' for i in range(n_customers)],
        'age': np.random.normal(45, 15, n_customers).astype(int),
        'gender': np.random.choice(['Male', 'Female'], n_customers),
        'tenure_months': np.random.exponential(20, n_customers).astype(int),
        'contract_type': np.random.choice(['month-to-month', 'one-year', 'two-year'], 
                                        n_customers, p=[0.5, 0.3, 0.2]),
        'internet_service': np.random.choice(['DSL', 'Fiber', 'None'], 
                                           n_customers, p=[0.4, 0.4, 0.2]),
        'tech_support': np.random.choice(['Yes', 'No'], n_customers, p=[0.3, 0.7]),
        'streaming_services': np.random.choice(['Yes', 'No'], n_customers, p=[0.4, 0.6]),
        'payment_method': np.random.choice(['Electronic check', 'Mailed check', 
                                          'Bank transfer', 'Credit card'], 
                                         n_customers, p=[0.35, 0.2, 0.2, 0.25])
    }
    
    # Clip age and tenure to realistic ranges
    data['age'] = np.clip(data['age'], 18, 80)
    data['tenure_months'] = np.clip(data['tenure_months'], 1, 72)
    
    # Generate monthly charges based on internet service type
    monthly_charges = []
    for service in data['internet_service']:
        if service == 'None':
            charge = np.random.normal(35, 10)
        elif service == 'DSL':
            charge = np.random.normal(55, 15)
        else:  # Fiber
            charge = np.random.normal(85, 20)
        monthly_charges.append(max(20, min(120, charge)))
    
    data['monthly_charges'] = np.round(monthly_charges, 2)
    
    # Generate total charges
    data['total_charges'] = np.round(
        np.array(data['tenure_months']) * np.array(data['monthly_charges']) + 
        np.random.normal(0, 100, n_customers), 2
    )
    data['total_charges'] = np.maximum(data['total_charges'], 0)
    
    # Generate churn with realistic relationships
    churn_probs = []
    for i in range(n_customers):
        prob = 0.15  # Base churn rate
        
        if data['contract_type'][i] == 'month-to-month':
            prob += 0.25
        elif data['contract_type'][i] == 'one-year':
            prob += 0.1
            
        if data['tenure_months'][i] < 6:
            prob += 0.2
        elif data['tenure_months'][i] < 12:
            prob += 0.1
        elif data['tenure_months'][i] > 24:
            prob -= 0.1
            
        if data['monthly_charges'][i] > 90:
            prob += 0.15
        elif data['monthly_charges'][i] < 30:
            prob += 0.1
            
        if data['tech_support'][i] == 'Yes':
            prob -= 0.08
            
        if data['payment_method'][i] == 'Electronic check':
            prob += 0.1
            
        if data['age'][i] < 25 or data['age'][i] > 65:
            prob += 0.05
            
        churn_probs.append(max(0.01, min(0.8, prob)))
    
    data['churn'] = np.random.binomial(1, churn_probs, n_customers)
    return pd.DataFrame(data)

# Generate data
telecom_data = generate_telecom_data(10000)
print(f"Data generated! Churn rate: {telecom_data['churn'].mean():.2%}")

# Create images directory
import os
if not os.path.exists('images'):
    os.makedirs('images')

# 1. CHURN DISTRIBUTION ANALYSIS
print("Creating churn distribution analysis...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Churn Analysis - Key Patterns', fontsize=16, y=1.02)

# Overall churn distribution
churn_counts = telecom_data['churn'].value_counts()
axes[0, 0].pie(churn_counts.values, labels=['Retained', 'Churned'], autopct='%1.1f%%', 
               colors=['lightblue', 'salmon'])
axes[0, 0].set_title('Overall Churn Distribution')

# Churn by contract type
contract_churn_pct = telecom_data.groupby('contract_type')['churn'].mean() * 100
contract_churn_pct.plot(kind='bar', ax=axes[0, 1], color='skyblue', rot=45)
axes[0, 1].set_title('Churn Rate by Contract Type')
axes[0, 1].set_ylabel('Churn Rate (%)')
axes[0, 1].grid(axis='y', alpha=0.3)

# Churn by tenure
telecom_data['tenure_group'] = pd.cut(telecom_data['tenure_months'], 
                                     bins=[0, 12, 24, 36, 72], 
                                     labels=['0-12', '13-24', '25-36', '37+'])
tenure_churn = telecom_data.groupby('tenure_group')['churn'].mean() * 100
tenure_churn.plot(kind='bar', ax=axes[0, 2], color='lightgreen', rot=0)
axes[0, 2].set_title('Churn Rate by Tenure Group (months)')
axes[0, 2].set_ylabel('Churn Rate (%)')
axes[0, 2].grid(axis='y', alpha=0.3)

# Monthly charges distribution
churned = telecom_data[telecom_data['churn'] == 1]['monthly_charges']
retained = telecom_data[telecom_data['churn'] == 0]['monthly_charges']
axes[1, 0].hist([retained, churned], bins=30, alpha=0.7, 
               label=['Retained', 'Churned'], color=['lightblue', 'salmon'])
axes[1, 0].set_title('Monthly Charges Distribution by Churn')
axes[1, 0].set_xlabel('Monthly Charges ($)')
axes[1, 0].set_ylabel('Frequency')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# Churn by internet service
internet_churn = telecom_data.groupby('internet_service')['churn'].mean() * 100
internet_churn.plot(kind='bar', ax=axes[1, 1], color='orange', rot=45)
axes[1, 1].set_title('Churn Rate by Internet Service')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].grid(axis='y', alpha=0.3)

# Churn by payment method
payment_churn = telecom_data.groupby('payment_method')['churn'].mean() * 100
payment_churn.plot(kind='bar', ax=axes[1, 2], color='pink', rot=45)
axes[1, 2].set_title('Churn Rate by Payment Method')
axes[1, 2].set_ylabel('Churn Rate (%)')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/churn_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. PREPARE DATA FOR MODELING
print("Preparing data for modeling...")
X = telecom_data.drop(['customer_id', 'churn', 'tenure_group'], axis=1)
y = telecom_data['churn']

# Encode categorical features
categorical_features = ['gender', 'contract_type', 'internet_service', 'tech_support', 
                       'streaming_services', 'payment_method']
numerical_features = ['age', 'tenure_months', 'monthly_charges', 'total_charges']

X_encoded = X.copy()
label_encoders = {}
for feature in categorical_features:
    le = LabelEncoder()
    X_encoded[feature] = le.fit_transform(X[feature])
    label_encoders[feature] = le

# Scale numerical features
scaler = StandardScaler()
X_scaled = X_encoded.copy()
X_scaled[numerical_features] = scaler.fit_transform(X_encoded[numerical_features])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# 3. TRAIN MODELS
print("Training models...")

# Logistic Regression
lr_model = LogisticRegression(random_state=42, max_iter=1000)
lr_model.fit(X_train, y_train)
y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]

# XGBoost (simplified for speed)
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                              n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
y_pred_proba_xgb = xgb_model.predict_proba(X_test)[:, 1]

# 4. ROC CURVE COMPARISON
print("Creating ROC curve comparison...")
fpr_lr, tpr_lr, _ = roc_curve(y_test, y_pred_proba_lr)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, y_pred_proba_xgb)
lr_auc = auc(fpr_lr, tpr_lr)
xgb_auc = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(10, 8))
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc:.3f})', 
         linewidth=2, color='blue')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc:.3f})', 
         linewidth=2, color='red')
plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random Classifier (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate', fontsize=12)
plt.ylabel('True Positive Rate', fontsize=12)
plt.title('ROC Curve Comparison - Churn Prediction Models', fontsize=14)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.savefig('images/roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. FEATURE IMPORTANCE
print("Creating feature importance analysis...")
feature_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
top_features = feature_importance.head(10)
plt.barh(range(len(top_features)), top_features['Importance'], color='skyblue')
plt.yticks(range(len(top_features)), top_features['Feature'])
plt.xlabel('Feature Importance')
plt.title('Top 10 Feature Importance (XGBoost)')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. BUSINESS IMPACT ANALYSIS
print("Creating business impact analysis...")
# Get predictions for test set
test_results = X_test.copy()
test_results['churn_probability'] = y_pred_proba_xgb
test_indices = y_test.index
test_results['monthly_charges'] = telecom_data.loc[test_indices, 'monthly_charges'].values

# Identify top 10% high-risk customers
top_10_percent = int(len(test_results) * 0.1)
high_risk_customers = test_results.nlargest(top_10_percent, 'churn_probability')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# Churn probability distribution
ax1.hist(test_results['churn_probability'], bins=30, alpha=0.7, color='skyblue', edgecolor='black')
ax1.axvline(high_risk_customers['churn_probability'].min(), color='red', linestyle='--', 
           label=f'Top 10% Threshold ({high_risk_customers["churn_probability"].min():.3f})')
ax1.set_xlabel('Churn Probability')
ax1.set_ylabel('Number of Customers')
ax1.set_title('Churn Probability Distribution')
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# Revenue impact simulation
high_risk_monthly_revenue = high_risk_customers['monthly_charges'].sum()
churners_in_high_risk = high_risk_customers[high_risk_customers.index.isin(
    y_test[y_test == 1].index)]
potential_revenue_at_risk = churners_in_high_risk['monthly_charges'].sum()
potential_savings = potential_revenue_at_risk * 0.4  # 40% retention success
net_benefit_monthly = potential_savings - (len(high_risk_customers) * 50 / 24)  # intervention cost

categories = ['Current\nMonthly Revenue', 'Revenue\nAt Risk', 'Potential\nSavings', 'Net Benefit\n(Monthly)']
values = [high_risk_monthly_revenue, potential_revenue_at_risk, potential_savings, net_benefit_monthly]

colors = ['lightblue', 'salmon', 'lightgreen', 'gold']
bars = ax2.bar(categories, values, color=colors, alpha=0.8)
ax2.set_ylabel('Revenue ($)')
ax2.set_title('Revenue Impact Analysis - High-Risk Customers')
ax2.grid(axis='y', alpha=0.3)

# Add value labels on bars
for bar, value in zip(bars, values):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
             f'${value:,.0f}', ha='center', va='bottom')

plt.tight_layout()
plt.savefig('images/business_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

print("All visualizations generated successfully!")
print("Generated files:")
print("- images/churn_distribution_analysis.png")
print("- images/roc_curve_comparison.png") 
print("- images/feature_importance.png")
print("- images/business_impact_analysis.png") 