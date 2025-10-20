import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset
import seaborn as sns


sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

def generate_training_data(n_samples=1000):
    """
    Simulate training data for a house price prediction model
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with training data
    """
    np.random.seed(42)
    
    data = {
        'square_feet': np.random.normal(2000, 500, n_samples),
        'bedrooms': np.random.randint(1, 6, n_samples),
        'age_years': np.random.normal(15, 10, n_samples),
        'distance_to_city': np.random.normal(10, 5, n_samples),
        'price': np.random.normal(350000, 100000, n_samples)
    }
    
    return pd.DataFrame(data)

def generate_drifted_data(n_samples=1000):
    """
    Simulate new data with drift (market changed over time)
    
    Args:
        n_samples: Number of samples to generate
    
    Returns:
        DataFrame with drifted data
    """
    np.random.seed(123)
    
    
    data = {
        'square_feet': np.random.normal(2200, 550, n_samples),  # Larger homes
        'bedrooms': np.random.randint(2, 7, n_samples),  # More bedrooms
        'age_years': np.random.normal(12, 8, n_samples),  # Newer homes
        'distance_to_city': np.random.normal(8, 4, n_samples),  # Closer to city
        'price': np.random.normal(450000, 120000, n_samples)  # Higher prices
    }
    
    return pd.DataFrame(data)

def plot_distribution_comparison(training_data, new_data, feature):
    """
    Plot side-by-side comparison of a feature's distribution
    
    Args:
        training_data: Original training dataset
        new_data: New incoming dataset
        feature: Column name to plot
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    
    axes[0].hist(training_data[feature], bins=35, alpha=0.8, color='blue', edgecolor='black')
    axes[0].set_title(f'Training Data - {feature}')
    axes[0].set_xlabel(feature)
    axes[0].set_ylabel('Frequency')
    axes[0].axvline(training_data[feature].mean(), color='red', linestyle='--', 
                    label=f'Mean: {training_data[feature].mean():.2f}')
    axes[0].legend()
    
    
    axes[1].hist(new_data[feature], bins=35, alpha=0.8, color='green', edgecolor='black')
    axes[1].set_title(f'New Data - {feature}')
    axes[1].set_xlabel(feature)
    axes[1].set_ylabel('Frequency')
    axes[1].axvline(new_data[feature].mean(), color='red', linestyle='--',
                    label=f'Mean: {new_data[feature].mean():.2f}')
    axes[1].legend()
    
    plt.tight_layout()
    plt.savefig(f'drift_comparison_{feature}.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"Plot saved as: drift_comparison_{feature}.png")

def calculate_drift_metrics(training_data, new_data):
    """
    Calculate simple drift metrics for all features
    
    Args:
        training_data: Original training dataset
        new_data: New incoming dataset
    
    Returns:
        DataFrame with drift statistics
    """
    metrics = []
    
    for column in training_data.columns:
        train_mean = training_data[column].mean()
        new_mean = new_data[column].mean()
        
        train_std = training_data[column].std()
        new_std = new_data[column].std()
        
        
        mean_change = ((new_mean - train_mean) / train_mean) * 100
        std_change = ((new_std - train_std) / train_std) * 100
        
    
        drift_detected = abs(mean_change) > 10
        
        metrics.append({
            'Feature': column,
            'Training Mean': f'{train_mean:.2f}',
            'New Mean': f'{new_mean:.2f}',
            'Mean Change %': f'{mean_change:.2f}%',
            'Std Change %': f'{std_change:.2f}%',
            'Drift Detected': 'YES' if drift_detected else 'NO'
        })
    
    return pd.DataFrame(metrics)

def main():
    """
    Main function to run drift detection analysis
    """
    print("=" * 60)
    print("DATA DRIFT DETECTION SYSTEM")
    print("=" * 60)
    print()
    
    
    print("Step 1: Generating training data...")
    training_data = generate_training_data(1000)
    print(f"Training data shape: {training_data.shape}")
    print(f"\nTraining data sample:\n{training_data.head()}")
    print()
    
    
    print("Step 2: Generating new data (simulating real-world changes)...")
    new_data = generate_drifted_data(1000)
    print(f"New data shape: {new_data.shape}")
    print(f"\nNew data sample:\n{new_data.head()}")
    print()
    
    
    print("Step 3: Calculating drift metrics...")
    drift_metrics = calculate_drift_metrics(training_data, new_data)
    print("\nDrift Analysis Results:")
    print(drift_metrics.to_string(index=False))
    print()
    
    
    print("Step 4: Creating visualizations...")
    plot_distribution_comparison(training_data, new_data, 'price')
    plot_distribution_comparison(training_data, new_data, 'square_feet')
    print()
    
    
    print("Step 5: Generating detailed Evidently report...")
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=training_data, current_data=new_data)
    report.save_html('detailed_drift_report.html')
    print("Detailed report saved as: detailed_drift_report.html")
    print()
    

    print("=" * 60)
    print("ANALYSIS COMPLETE!")
    print("=" * 60)
    print("\nFiles generated:")
    print("1. drift_comparison_price.png - Price distribution comparison")
    print("2. drift_comparison_square_feet.png - Square feet comparison")
    print("3. detailed_drift_report.html - Full interactive report")
    print("\nRecommendation:")
    drift_count = drift_metrics[drift_metrics['Drift Detected'] == 'YES'].shape[0]
    if drift_count > 0:
        print(f"WARNING  DRIFT DETECTED in {drift_count} feature(s)!")
        print("   Action needed: Consider retraining the model with recent data.")
    else:
        print("SUCCESS No significant drift detected. Model is still reliable.")

if __name__ == "__main__":
    main()
