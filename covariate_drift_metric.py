# covariate_drift_metric.py
import pandas as pd
from scipy.stats import ks_2samp

from a4s_eval.data_model.evaluation import Dataset, DataShape, FeatureType
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.data_metric_registry import data_metric


def calculate_distribution_change(x_ref: pd.Series, x_new: pd.Series) -> float:
    """Calculate how much the distribution changed between reference and new data.
    
    Returns a weight based on mean and variance differences.
    """
    # How much did the mean change?
    mean_change = abs(x_new.mean() - x_ref.mean()) / (abs(x_ref.mean()) + 0.0001)
    
    # How much did the variance change?
    var_change = abs(x_new.var() - x_ref.var()) / (abs(x_ref.var()) + 0.0001)
    
    # Combine both changes
    weight = (mean_change**2 + var_change**2) ** 0.5
    
    return float(weight)


def covariate_drift_score(x_ref: pd.Series, x_new: pd.Series) -> float:
    """Calculate covariate drift using KS test weighted by distribution change."""
    
    # Use KS test to compare distributions
    ks_statistic, _ = ks_2samp(x_ref, x_new)
    
    # Calculate weight based on distribution change
    weight = calculate_distribution_change(x_ref, x_new)
    
    # Multiply KS statistic by weight (DFKI method)
    drift_score = ks_statistic * (1 + weight)
    
    return float(drift_score)


@data_metric(name="Covariate drift (DFKI)")
def covariate_drift_metric(
    datashape: DataShape,
    reference: Dataset,
    evaluated: Dataset,
) -> list[Measure]:
    """Calculate covariate drift for numerical features.
    
    Compares reference dataset to evaluated dataset using gradient-based weighting.
    """
    
    # Get the date from evaluated dataset
    date_column = datashape.date.name
    current_date = pd.to_datetime(evaluated.data[date_column]).max()
    
    # Get feature mappings from evaluated dataset
    feature_mapping = {
        feature.name: feature.pid 
        for feature in evaluated.shape.features
    }
    
    metrics = []
    
    # Loop through each feature in the datashape
    for feature in datashape.features:
        
        # Only process numerical features
        if feature.feature_type not in [FeatureType.INTEGER, FeatureType.FLOAT]:
            continue
        
        # Get feature data from both datasets
        ref_data = reference.data[feature.name]
        eval_data = evaluated.data[feature.name]
        
        # Calculate drift score
        score = covariate_drift_score(ref_data, eval_data)
        
        # Create metric
        metric = Measure(
            name="covariate_drift_dfki",
            score=score,
            time=current_date.to_pydatetime(),
            feature_pid=feature_mapping.get(feature.name),
        )
        
        metrics.append(metric)
    
    return metrics