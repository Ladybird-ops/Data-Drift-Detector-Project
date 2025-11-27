# test_covariate_drift_metric.py
"""

"""
import uuid
import numpy as np
import pandas as pd
import pytest

from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.metrics.data_metrics.covariate_drift_metric import covariate_drift_metric


@pytest.fixture
def data_shape() -> DataShape:
    """Load and prepare the data shape for testing."""
    metadata = pd.read_csv("tests/data/lcld_v2_metadata_api.csv").to_dict(
        orient="records"
    )

    # Add unique IDs to each feature
    for record in metadata:
        record["pid"] = uuid.uuid4()

    # Build data shape structure
    data_shape = {
        "features": [
            item
            for item in metadata
            if item.get("name") not in ["charged_off", "issue_d"]
        ],
        "target": next(rec for rec in metadata if rec.get("name") == "charged_off"),
        "date": next(rec for rec in metadata if rec.get("name") == "issue_d"),
    }

    return DataShape.model_validate(data_shape)


@pytest.fixture
def test_dataset(tab_class_test_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    """Prepare test dataset."""
    data = tab_class_test_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


@pytest.fixture
def ref_dataset(tab_class_train_data: pd.DataFrame, data_shape: DataShape) -> Dataset:
    """Prepare reference dataset."""
    data = tab_class_train_data
    data["issue_d"] = pd.to_datetime(data["issue_d"])
    return Dataset(pid=uuid.uuid4(), shape=data_shape, data=data)


def test_covariate_drift_generates_metrics(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    """
    Test that covariate drift metric generates metrics for numerical features.
    This is the main execution test for the covariate drift metric.
    """
    
    # Execute the covariate drift metric
    metrics = covariate_drift_metric(data_shape, ref_dataset, test_dataset)
    
    # Count numerical features in the dataset
    numerical_count = sum(
        1 for f in data_shape.features 
        if f.feature_type.name in ["INTEGER", "FLOAT"]
    )
    
    # Verify: should have one metric per numerical feature
    assert len(metrics) == numerical_count
    print(f"✓ Generated {len(metrics)} covariate drift metrics")


def test_covariate_drift_scores_are_valid(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    """
    Test that all covariate drift scores are valid numbers (not NaN, positive).
    """
    
    # Execute the covariate drift metric
    metrics = covariate_drift_metric(data_shape, ref_dataset, test_dataset)
    
    # Verify: no NaN values
    assert all(not np.isnan(metric.score) for metric in metrics)
    
    # Verify: all scores should be non-negative
    assert all(metric.score >= 0 for metric in metrics)
    
    print(f"✓ All {len(metrics)} scores are valid")
    for metric in metrics[:3]:  # Print first 3 as examples
        print(f"  - Score: {metric.score:.4f}")


def test_covariate_drift_has_feature_pids(
    data_shape: DataShape, ref_dataset: Dataset, test_dataset: Dataset
):
    """
    Test that each metric has a feature PID assigned.
    """
    
    # Execute the covariate drift metric
    metrics = covariate_drift_metric(data_shape, ref_dataset, test_dataset)
    
    # Verify: all metrics have feature PIDs
    assert all(metric.feature_pid is not None for metric in metrics)
    
    print(f"✓ All {len(metrics)} metrics have feature PIDs assigned")


if __name__ == "__main__":
    """
    Direct execution support for quick testing.
    Run: python test_covariate_drift_metric.py
    """
    pytest.main([__file__, "-v", "-s"])