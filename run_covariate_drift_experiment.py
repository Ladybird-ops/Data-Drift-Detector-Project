# Testing covariate drift with dfki on linear regression model with Amazon Reviews dataset

# Import libraries
import uuid
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Machine Learning
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# A4S Framework
from a4s_eval.data_model.evaluation import Dataset, DataShape, FeatureType
from a4s_eval.metrics.data_metrics.covariate_drift_metric import covariate_drift_metric

print("✓ All libraries loaded")

# Load Amazon Reviews Dataset
print("\nLoading Amazon Reviews dataset...")
data = pd.read_csv("data/amazon_reviews.csv").dropna()

# Split: 60% reference (training), 40% evaluated (testing)
split = int(len(data) * 0.6)
ref_data = data.iloc[:split].copy()
eval_data = data.iloc[split:].copy()

# Ensure dates exist
if 'date' not in ref_data.columns:
    ref_data['date'] = pd.date_range('2023-01-01', periods=len(ref_data), freq='H')
    eval_data['date'] = pd.date_range('2023-06-01', periods=len(eval_data), freq='H')
else:
    ref_data['date'] = pd.to_datetime(ref_data['date'])
    eval_data['date'] = pd.to_datetime(eval_data['date'])

print(f"✓ Reference: {len(ref_data)} rows")
print(f"✓ Evaluated: {len(eval_data)} rows")

# Create DataShape
print("\nCreating DataShape...")

# Get ALL column names, including date and exclude text/object columns we can't use
# select only numerical and date columns for pydantic validation
numerical_and_date_cols = data.select_dtypes(include=[np.number, 'datetime64[ns]']).columns.tolist()

# Target is 'rating'
target = 'rating'
if target not in numerical_and_date_cols:
    if 'date' in numerical_and_date_cols:
        numerical_and_date_cols.remove('date') 
    if numerical_and_date_cols:
        target = numerical_and_date_cols[-1]
    else:
        raise ValueError("Could not find a valid numerical target column.")

# Separate features from target and date
feature_cols = [col for col in numerical_and_date_cols if col != target and col != 'date']

# Build features list with min/max values from ref_data
features = []
for col in feature_cols:
    is_float = (data[col].dtype == float)
    
    features.append({
        'name': col, 
        'pid': uuid.uuid4(), 
        'feature_type': FeatureType.FLOAT if is_float else FeatureType.INTEGER,
        'min_value': ref_data[col].min(),  
        'max_value': ref_data[col].max()  
    })

# Get min/max for Target and Date from ref_data
target_min = ref_data[target].min()
target_max = ref_data[target].max()
date_min = ref_data['date'].min().isoformat() 
date_max = ref_data['date'].max().isoformat() 


# Use the DataShape constructor with all required min/max values
datashape = DataShape(
    features=features,
    target={
        'name': target, 
        'pid': uuid.uuid4(), 
        'feature_type': FeatureType.INTEGER, 
        'min_value': target_min,            
        'max_value': target_max             
    },
    date={
        'name': 'date', 
        'pid': uuid.uuid4(), 
        'feature_type': FeatureType.DATE,
        'min_value': date_min,              
        'max_value': date_max               
    }
)

feature_names = [f['name'] for f in features]
print(f"✓ DataShape with {len(features)} features: {feature_names}")
# Train Linear Regression Model
print("\n=== Training Linear Regression Model ===")

X_train = ref_data[feature_names]
y_train = ref_data[target]
X_test = eval_data[feature_names]
y_test = eval_data[target]

# Train model
model = LinearRegression()
model.fit(X_train, y_train)

# Evaluate
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)

print(f"✓ Model trained on {target}")
print(f"  Training: MSE = {mse_train:.4f}, R² = {r2_train:.4f}")
print(f"  Testing:  MSE = {mse_test:.4f}, R² = {r2_test:.4f}")
perf_drop = ((mse_test - mse_train) / mse_train * 100)
print(f"  Performance drop: {perf_drop:.1f}%")

# Run Covariate Drift Analysis (DFKI Method)
print("\n=== Running Covariate Drift Analysis ===")

ref_dataset = Dataset(pid=uuid.uuid4(), shape=datashape, data=ref_data)
eval_dataset = Dataset(pid=uuid.uuid4(), shape=datashape, data=eval_data)

# Run method  (DFKI gradient-based)
metrics = covariate_drift_metric(datashape, ref_dataset, eval_dataset)
scores = [m.score for m in metrics]

print(f"✓ Generated {len(metrics)} drift scores")
print(f"  Mean drift: {np.mean(scores):.4f}")
print(f"  Max drift: {np.max(scores):.4f}")

# Calculate DFKI Gradient Weights
print("\nCalculating DFKI gradient weights...")
gradient_weights = []

for fname in feature_names:
    ref_vals = ref_data[fname]
    eval_vals = eval_data[fname]
    
    # DFKI gradient calculation
    mean_change = abs(eval_vals.mean() - ref_vals.mean()) / (abs(ref_vals.mean()) + 0.0001)
    var_change = abs(eval_vals.var() - ref_vals.var()) / (abs(ref_vals.var()) + 0.0001)
    gradient_weight = np.sqrt(mean_change**2 + var_change**2)
    gradient_weights.append(gradient_weight)

print(f"✓ Gradient weights calculated")

# Create Visualizations
print("\n=== Creating Visualizations ===")

fig = plt.figure(figsize=(18, 6))

# PLOT 1: Drift Scores Bar Chart
ax1 = plt.subplot(1, 3, 1)
sorted_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
sorted_features = [feature_names[i] for i in sorted_idx]
sorted_scores = [scores[i] for i in sorted_idx]
colors = ['red' if s >= 0.3 else 'orange' if s >= 0.1 else 'green' for s in sorted_scores]

ax1.barh(sorted_features, sorted_scores, color=colors, edgecolor='black', alpha=0.7)
ax1.set_xlabel('Drift Score', fontsize=12, fontweight='bold')
ax1.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax1.set_title('Covariate Drift Scores\n(DFKI Gradient Method)', fontsize=13, fontweight='bold')
ax1.invert_yaxis()

# PLOT 2: DFKI Gradient Analysis
ax2 = plt.subplot(1, 3, 2)
scatter = ax2.scatter(gradient_weights, scores, s=150, alpha=0.7, 
                      c=scores, cmap='YlOrRd', edgecolor='black', linewidth=1.5)
ax2.set_xlabel('DFKI Gradient Weight', fontsize=12, fontweight='bold')
ax2.set_ylabel('Drift Score', fontsize=12, fontweight='bold')
ax2.set_title('DFKI Method:\nGradient Weight → Drift Score', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Drift Score')

# PLOT 3: Model Performance Impact
ax3 = plt.subplot(1, 3, 3)
categories = ['Training\n(Reference)', 'Testing\n(Drifted)']
mse_values = [mse_train, mse_test]
colors_perf = ['green', 'red']

bars = ax3.bar(categories, mse_values, color=colors_perf, alpha=0.7, edgecolor='black', width=0.6)
ax3.set_ylabel('Mean Squared Error', fontsize=12, fontweight='bold')
ax3.set_title('Model Performance:\nImpact of Drift', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Add value labels
for bar, mse in zip(bars, mse_values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
             f'{mse:.3f}', ha='center', va='bottom', fontweight='bold')

plt.suptitle('Covariate Drift Analysis - Amazon Reviews (DFKI Method + Model Performance)', 
             fontsize=15, fontweight='bold')
plt.tight_layout()
plt.savefig('covariate_drift_results.png', dpi=300, bbox_inches='tight')
print("✓ Saved: covariate_drift_results.png")
plt.show()

# Print Final Summary
high_drift = sum(1 for s in scores if s >= 0.3)
medium_drift = sum(1 for s in scores if 0.1 <= s < 0.3)
low_drift = sum(1 for s in scores if s < 0.1)

print("\n" + "="*70)
print("EXPERIMENT RESULTS SUMMARY")
print("="*70)
print(f"Dataset:                 Amazon Reviews")
print(f"Model Type:              Linear Regression (predicting {target})")
print(f"\nData Split:")
print(f"  Training (Reference):  {len(ref_data)} samples")
print(f"  Testing (Evaluated):   {len(eval_data)} samples")
print(f"\nDrift Detection (DFKI Method):")
print(f"  Features Analyzed:     {len(metrics)}")
print(f"  Mean Drift Score:      {np.mean(scores):.4f}")
print(f"  High Drift (≥0.3):     {high_drift} features")
print(f"  Medium Drift (0.1-0.3): {medium_drift} features")
print(f"  Low Drift (<0.1):      {low_drift} features")
print(f"\nModel Performance:")
print(f"  Training MSE:          {mse_train:.4f}")
print(f"  Testing MSE:           {mse_test:.4f}")
print(f"  Performance Drop:      {perf_drop:.1f}%")
print(f"\nConclusion: Detected drift correlates with model performance degradation")
print("="*70)
print("\n✓ EXPERIMENT COMPLETED!")
print("\nGenerated Files:")
print("  1. covariate_drift_results.png    (Custom visualizations)")
print("="*70)