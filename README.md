1.	 Metric Information
Metric name: Covariate Drift (DFKI Method)
What it does: This metric detects distribution changes in numerical features between two datasets (reference and evaluated). It identifies which and how much the features have shifted over time. 

2.	Method Used
•	KS Test: Measures how different two distributions are
•	Gradient Weighting: Calculates how much the mean and variance changed
•	Final Score: KS statistic × (1 + gradient_weight)

Higher scores mean more drift detected.
3.	Models It Applies To: This metric is model agnostic as it applies to all models. Also, because it is a data metric, it does not really need a model to run.

4.	Data types it applies to: this metric applies to numerical features only. That means intergers and floats. It also requres that the reference and evaluated datasets are of the same structure.

5.	Key Assumptions
•	Both datasets have the same features
•	Features contain valid numerical values
•	A date column exists for timestamping results

6.	 Test Implementation Details: This metric was implemented within the a4s-eval environment and based on its framework.
Metric implementation file location:
a4s_eval/metrics/data_metrics/covariate_drift_metric.py

Test File:
tests/metrics/data_metrics/test_covariate_drift_metric.py

How to Run the Test: the test can be ran using this command:
uv run pytest -s tests/metrics/data_metrics/test_covariate_drift_metric.py

Expected result:
collected 3 items
✓ Generated 23 covariate drift metrics
✓ All 23 scores are valid
✓ All 23 metrics have feature PIDs assigned
3 passed in ~24s

Dataset Used
Dataset Name: Lending Club Loan Data (LCLD v2)

Source: Built into the a4s-eval framework
•	No manual download required
•	Automatically available through framework fixtures
•	Reference data: tab_class_train_data
•	Evaluated data: tab_class_test_data
•	Metadata: tests/data/lcld_v2_metadata_api.csv

Dataset Characteristics:
•	23 numerical features
•	Date column: issue_d
•	Target column: charged_off
•	Contains loan information (amounts, rates, terms, etc.)

Test Functions
The test file contains 3 test functions that verify the metric works correctly:

a.	 test_covariate_drift_generates_metrics
•	Verifies that metrics are created for all numerical features
•	Checks that 23 metrics are generated

b.	 test_covariate_drift_scores_are_valid
•	Ensures all scores are valid numbers 
•	Confirms all scores are non-negative

c.	 test_covariate_drift_has_feature_pids
•	Verifies each metric has a feature ID assigned
•	Ensures proper metadata linking

Parameters: No manual parameters required.
The metric automatically:
•	Identifies numerical features from the dataset
•	Computes drift scores for each feature
•	Assigns feature IDs
•	Adds timestamps

Everything runs automatically when you call the metric function.

Test Results
All tests passed successfully:
•	✅ 3 out of 3 tests passed
•	✅ 23 drift scores generated
•	✅ Sample scores: 0.0800, 0.0227, 0.4501
•	✅ All scores are valid numbers
•	✅ All feature IDs properly assigned
The metric successfully detected varying levels of drift across different features in the dataset.

Summary
This implementation provides a working covariate drift metric that:
•	Successfully detects distribution changes in numerical features
•	Uses the DFKI gradient-based weighting method
•	Works with any tabular dataset containing numerical features
•	Requires no manual configuration
•	Has been tested and verified with all tests passing
•	Integrates properly with the a4s-eval framework
