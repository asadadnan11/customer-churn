#!/usr/bin/env python3
"""
Generate some nice charts for the README
Quick script to make the visualizations we need
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
warnings.filterwarnings('ignore')  # don't want to see warnings

# setup plotting style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")
np.random.seed(42)

print("Making some charts for the README...")

def generate_telecom_data(num_customers=10000):
    """Make fake telecom data that looks realistic"""
    print(f"Creating {num_customers:,} fake customers...")
    
    # build the data step by step
    data = {}
    
    # customer IDs
    data['customer_id'] = [f'CUST_{i+1:06d}' for i in range(num_customers)]
    
    # demographics
    data['age'] = np.random.normal(45, 15, num_customers).astype(int)
    data['gender'] = np.random.choice(['Male', 'Female'], num_customers)
    
    # service info
    data['tenure_months'] = np.random.exponential(20, num_customers).astype(int)
    data['contract_type'] = np.random.choice(['month-to-month', 'one-year', 'two-year'], 
                                           num_customers, p=[0.5, 0.3, 0.2])
    data['internet_service'] = np.random.choice(['DSL', 'Fiber', 'None'], 
                                              num_customers, p=[0.4, 0.4, 0.2])
    data['tech_support'] = np.random.choice(['Yes', 'No'], num_customers, p=[0.3, 0.7])
    data['streaming_services'] = np.random.choice(['Yes', 'No'], num_customers, p=[0.4, 0.6])
    
    # payment method
    payment_options = ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card']
    data['payment_method'] = np.random.choice(payment_options, 
                                            num_customers, p=[0.35, 0.2, 0.2, 0.25])
    
    # fix ranges
    data['age'] = np.clip(data['age'], 18, 80)
    data['tenure_months'] = np.clip(data['tenure_months'], 1, 72)
    
    # monthly charges depend on service type
    charges_list = []
    for service_type in data['internet_service']:
        if service_type == 'None':
            charge = np.random.normal(35, 10)
        elif service_type == 'DSL':
            charge = np.random.normal(55, 15)
        else:  # Fiber
            charge = np.random.normal(85, 20)
        charges_list.append(max(20, min(120, charge)))
    
    data['monthly_charges'] = [round(x, 2) for x in charges_list]
    
    # total charges calculation
    total_charges_calc = []
    for i in range(num_customers):
        base = data['tenure_months'][i] * data['monthly_charges'][i]
        noise = np.random.normal(0, 100)
        total = base + noise
        total_charges_calc.append(max(0, round(total, 2)))
    
    data['total_charges'] = total_charges_calc
    
    # now generate churn - this is the important part
    churn_probabilities = []
    for i in range(num_customers):
        prob = 0.15  # baseline
        
        # contract type matters a lot
        if data['contract_type'][i] == 'month-to-month':
            prob += 0.25
        elif data['contract_type'][i] == 'one-year':
            prob += 0.1
            
        # tenure effect
        if data['tenure_months'][i] < 6:
            prob += 0.2
        elif data['tenure_months'][i] < 12:
            prob += 0.1
        elif data['tenure_months'][i] > 24:
            prob -= 0.1
            
        # price sensitivity
        if data['monthly_charges'][i] > 90:
            prob += 0.15
        elif data['monthly_charges'][i] < 30:
            prob += 0.1
            
        # support helps
        if data['tech_support'][i] == 'Yes':
            prob -= 0.08
            
        # payment method
        if data['payment_method'][i] == 'Electronic check':
            prob += 0.1
            
        # age patterns
        if data['age'][i] < 25 or data['age'][i] > 65:
            prob += 0.05
            
        prob = max(0.01, min(0.8, prob))
        churn_probabilities.append(prob)
    
    data['churn'] = np.random.binomial(1, churn_probabilities, num_customers)
    return pd.DataFrame(data)

# create our dataset
dataset = generate_telecom_data(10000)
print(f"Done! Churn rate: {dataset['churn'].mean():.1%}")

# make sure images folder exists
import os
if not os.path.exists('images'):
    os.makedirs('images')
    print("Created images directory")

# CHART 1: Main churn analysis dashboard
print("Making the main churn analysis chart...")
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
fig.suptitle('Customer Churn Patterns - Visual Analysis', fontsize=16, y=1.02)

# overall pie chart
churn_counts = dataset['churn'].value_counts()
axes[0, 0].pie(churn_counts.values, labels=['Stayed', 'Left'], autopct='%1.1f%%', 
               colors=['lightgreen', 'lightcoral'])
axes[0, 0].set_title('Overall: Who Stayed vs Left')

# contract type - should show big differences
contract_rates = dataset.groupby('contract_type')['churn'].mean() * 100
contract_rates.plot(kind='bar', ax=axes[0, 1], color='steelblue', rot=45)
axes[0, 1].set_title('Churn % by Contract Length')
axes[0, 1].set_ylabel('Churn Rate (%)')
axes[0, 1].grid(axis='y', alpha=0.3)

# tenure groups
dataset_copy = dataset.copy()
dataset_copy['tenure_group'] = pd.cut(dataset_copy['tenure_months'], 
                                     bins=[0, 12, 24, 36, 72], 
                                     labels=['0-12 mo', '13-24 mo', '25-36 mo', '37+ mo'])
tenure_rates = dataset_copy.groupby('tenure_group')['churn'].mean() * 100
tenure_rates.plot(kind='bar', ax=axes[0, 2], color='orange', rot=0)
axes[0, 2].set_title('Churn by Customer Tenure')
axes[0, 2].set_ylabel('Churn Rate (%)')
axes[0, 2].grid(axis='y', alpha=0.3)

# monthly charges histogram
churned_customers = dataset[dataset['churn'] == 1]['monthly_charges']
staying_customers = dataset[dataset['churn'] == 0]['monthly_charges']
axes[1, 0].hist([staying_customers, churned_customers], bins=25, alpha=0.6, 
               label=['Stayed', 'Left'], color=['green', 'red'])
axes[1, 0].set_title('Monthly Bill: Stayers vs Leavers')
axes[1, 0].set_xlabel('Monthly Charges ($)')
axes[1, 0].set_ylabel('Number of Customers')
axes[1, 0].legend()
axes[1, 0].grid(axis='y', alpha=0.3)

# internet service impact
internet_rates = dataset.groupby('internet_service')['churn'].mean() * 100
internet_rates.plot(kind='bar', ax=axes[1, 1], color='purple', rot=30)
axes[1, 1].set_title('Internet Service vs Churn')
axes[1, 1].set_ylabel('Churn Rate (%)')
axes[1, 1].grid(axis='y', alpha=0.3)

# payment method
payment_rates = dataset.groupby('payment_method')['churn'].mean() * 100
payment_rates.plot(kind='bar', ax=axes[1, 2], color='teal', rot=30)
axes[1, 2].set_title('Payment Method Impact')
axes[1, 2].set_ylabel('Churn Rate (%)')
axes[1, 2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('images/churn_distribution_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved churn distribution analysis")

# now prepare data for machine learning
print("Getting data ready for ML models...")
features = dataset.drop(['customer_id', 'churn'], axis=1)
target = dataset['churn']

# encode categorical variables
cat_cols = ['gender', 'contract_type', 'internet_service', 'tech_support', 
           'streaming_services', 'payment_method']
num_cols = ['age', 'tenure_months', 'monthly_charges', 'total_charges']

# do the encoding
features_encoded = features.copy()
encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    features_encoded[col] = le.fit_transform(features[col])
    encoders[col] = le

# scale the numeric features
scaler = StandardScaler()
features_scaled = features_encoded.copy()
features_scaled[num_cols] = scaler.fit_transform(features_encoded[num_cols])

# split the data
X_train, X_test, y_train, y_test = train_test_split(
    features_scaled, target, test_size=0.2, random_state=42, stratify=target
)

# build the models
print("Training logistic regression...")
# start with simple logistic regression
log_reg = LogisticRegression(random_state=42, max_iter=1000)
log_reg.fit(X_train, y_train)
lr_probs = log_reg.predict_proba(X_test)[:, 1]

print("Training XGBoost (using simple params for speed)...")
# xgboost with basic settings - don't need fancy hyperparameter tuning for this
xgb_model = xgb.XGBClassifier(random_state=42, eval_metric='logloss',
                             n_estimators=100, max_depth=5, learning_rate=0.1)
xgb_model.fit(X_train, y_train)
xgb_probs = xgb_model.predict_proba(X_test)[:, 1]

# CHART 2: ROC curves 
print("Making ROC curve comparison chart...")
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
fpr_xgb, tpr_xgb, _ = roc_curve(y_test, xgb_probs)
lr_auc_score = auc(fpr_lr, tpr_lr)
xgb_auc_score = auc(fpr_xgb, tpr_xgb)

plt.figure(figsize=(10, 8))
# plot both curves
plt.plot(fpr_lr, tpr_lr, label=f'Logistic Regression (AUC = {lr_auc_score:.3f})', 
         linewidth=2.5, color='blue')
plt.plot(fpr_xgb, tpr_xgb, label=f'XGBoost (AUC = {xgb_auc_score:.3f})', 
         linewidth=2.5, color='red')
# add random line for reference  
plt.plot([0, 1], [0, 1], 'k--', alpha=0.6, label='Random Guessing (AUC = 0.500)')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
plt.title('Model Performance Comparison: ROC Curves', fontsize=14, pad=20)
plt.legend(loc="lower right", fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('images/roc_curve_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved ROC curve comparison")

# CHART 3: what features matter most?
print("Making feature importance chart...")
# get feature importance from xgboost model
importance_data = pd.DataFrame({
    'Feature': X_train.columns,
    'Importance': xgb_model.feature_importances_
}).sort_values('Importance', ascending=False)

plt.figure(figsize=(12, 8))
# just show top 10 to keep it clean
top_10_features = importance_data.head(10)
plt.barh(range(len(top_10_features)), top_10_features['Importance'], color='lightcoral')
plt.yticks(range(len(top_10_features)), top_10_features['Feature'])
plt.xlabel('Importance Score', fontsize=12)
plt.title('What Features Matter Most for Predicting Churn?', fontsize=14, pad=20)
plt.gca().invert_yaxis()  # flip so most important is at top
plt.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig('images/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved feature importance chart")

# CHART 4: business impact - the money stuff
print("Making business impact chart...")
# calculate business metrics using our test data
test_data = X_test.copy()
test_data['churn_prob'] = xgb_probs
test_indices = y_test.index
test_data['monthly_bill'] = dataset.loc[test_indices, 'monthly_charges'].values

# find the riskiest 10% of customers
num_high_risk = int(len(test_data) * 0.1)
risky_customers = test_data.nlargest(num_high_risk, 'churn_prob')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

# probability distribution chart
ax1.hist(test_data['churn_prob'], bins=25, alpha=0.7, color='steelblue', edgecolor='black')
threshold = risky_customers['churn_prob'].min()
ax1.axvline(threshold, color='red', linestyle='--', linewidth=2,
           label=f'High Risk Cutoff ({threshold:.3f})')
ax1.set_xlabel('Predicted Churn Probability', fontsize=12)
ax1.set_ylabel('Number of Customers', fontsize=12)
ax1.set_title('Distribution of Churn Risk Scores', fontsize=13)
ax1.legend()
ax1.grid(axis='y', alpha=0.3)

# revenue impact calculations (making some assumptions here)
total_revenue_at_risk = risky_customers['monthly_bill'].sum()
# find actual churners in our high risk group
actual_churners = risky_customers[risky_customers.index.isin(y_test[y_test == 1].index)]
real_revenue_loss = actual_churners['monthly_bill'].sum()
# assume we can save 40% with intervention
savings_potential = real_revenue_loss * 0.4  
# intervention costs around $50 per customer (spread over 2 years)
intervention_cost = len(risky_customers) * 50 / 24
net_monthly_benefit = savings_potential - intervention_cost

# make the revenue chart
categories = ['High Risk\nRevenue', 'Actual\nLoss Risk', 'Potential\nSavings', 'Net Monthly\nBenefit']
amounts = [total_revenue_at_risk, real_revenue_loss, savings_potential, net_monthly_benefit]
colors = ['lightblue', 'lightcoral', 'lightgreen', 'gold']

bars = ax2.bar(categories, amounts, color=colors, alpha=0.8)
ax2.set_ylabel('Monthly Revenue ($)', fontsize=12)
ax2.set_title('Business Impact: Revenue Protection Opportunity', fontsize=13)
ax2.grid(axis='y', alpha=0.3)

# add dollar amounts on the bars
for bar, amount in zip(bars, amounts):
    height = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.02,
             f'${amount:,.0f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('images/business_impact_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print("Saved business impact analysis")

print("\nAll done! Generated these visualization files:")
print("✓ images/churn_distribution_analysis.png")
print("✓ images/roc_curve_comparison.png") 
print("✓ images/feature_importance.png")
print("✓ images/business_impact_analysis.png")
print("\nReady to update the README!") 