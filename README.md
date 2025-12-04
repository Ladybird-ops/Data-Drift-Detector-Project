# Covariate Drift Detection with DFKI Gradient Based Method

**GitHub Repository:** https://github.com/Ladybird-ops/Data-Drift-Detector-Project/tree/main

## Project Overview
This project implements a **Covariate Drift Detection Metric** using the DFKI gradient-based weighting method. The experiment demonstrates how changes in input data distribution over time, affect machine learning model performance.


**Dataset:** Amazon Reviews  
**Model:** Linear Regression (predicting ratings)  
**Metric:** Covariate Drift Detection with DFKI gradient-based method  

## Repository Structure
```
a4s-eval/
├── a4s_eval/
│   └── metrics/
│       └── data_metrics/
│           └── covariate_drift_metric.py    # the Metric implementation
├── tests/
│   └── metrics/
│       └── data_metrics/
│           └── test_covariate_drift_metric.py  # the Unit tests
├── data/
│   └── amazon_reviews.csv                   # Amazon Reviews dataset
├── run_covariate_drift_experiment.py        # Main experiment script
├── requirements.txt                         # Python dependencies
└── README.md                                
```

## Installation & Setup

### Prerequisites
- Python 3.12.3
- A4S evaluation environment (already configured)

### Setup Steps
```bash
# Extract the archive
unzip covariate_drift_submission.zip
cd a4s-eval

# Create and activate virtual environment
source .venv/bin/activate #for windows system

# Install dependencies
## Dependencies

Listed in `requirements.txt`:
- pandas 
- numpy 
- scipy 
- matplotlib 
- seaborn
- scikit-learn 

Install with:
```bash
uv pip install -r requirements.txt
```

## The Unit Tests

This project includes comprehensive unit tests for the covariate drift metric implementation.

### To run the tests
```bash
uv run pytest -s tests/metrics/data_metrics/test_covariate_drift_metric.py
```

### Expected Output
========== test session starts ==========
platform linux -- Python 3.12.3, pytest-8.4.1, pluggy-1.6.0
rootdir: /home/chiamaka1997/a4s/a4s-eval
configfile: pyproject.toml
plugins: cov-6.2.1, anyio-4.9.0
collected 3 items

tests/metrics/data_metrics/test_covariate_drift_metric.py ✓ Generated 23 covariate drift metrics
✓ All 23 scores are valid
  - Score: 0.0800
  - Score: 0.0227
  - Score: 0.4501
✓ All 23 metrics have feature PIDs assigned

========== 3 passed in 24.31s ==========

### what the tests verify
- Generates 23 covariate drift metrics successfully
- All drift scores are valid (between 0 and 1)
- All metrics have proper feature identifiers (PIDs) assigned
- DFKI gradient-based calculation works correctly

### Running the Experiment

### To execute complete experiment use this commmand
```bash
python run_covariate_drift_experiment.py
```

### What the Experiment Does
1. Loads Amazon Reviews dataset (1,000 reviews)
2. Splits data: 60% reference (training), 40% evaluation (testing)
3. Trains Linear Regression model to predict ratings
4. Calculates covariate drift scores using DFKI method
5. Generates visualization
6. Displays comprehensive results summary

### Generated Output
**File created:**
- `covariate_drift_results.png` - Visualization showing drift analysis

## Experiment Results summary

```
✓ Reference: 600 samples (older reviews)
✓ Evaluated: 400 samples (newer reviews)

✓ DataShape with 5 features: ['sentiment', 'review_length', 'word_count', 
  'helpfulness_score', 'verified_purchase']

=== Model Performance ===
Training: MSE = 0.4397, R² = 0.6089
Testing:  MSE = 0.4651, R² = 0.5800
Performance drop: 5.8%

=== Covariate Drift Analysis ===
Mean drift: 0.3013
Max drift:  0.8898

High Drift (≥0.3):    2 features
Medium Drift (0.1-0.3): 1 feature
Low Drift (<0.1):     2 features

Conclusion: Detected drift correlates with model performance degradation
```

### Key Findings
- **Moderate drift detected** (mean score: 0.3013)
- **2 features** show significant distribution changes
- **Model performance dropped 5.8%** from training to testing
- Drift in features correlates with performance degradation

## Dataset Information

### Amazon Reviews Dataset
**Features analyzed:**
1. **sentiment** - Review sentiment score
2. **review_length** - Number of characters
3. **word_count** - Number of words
4. **helpfulness_score** - Review helpfulness rating
5. **verified_purchase** - Purchase verification status


## Metric Implementation

### DFKI Gradient-Based Method

The covariate drift metric measures how much your data has changed over time using two main components:

**1. KS Test (Kolmogorov-Smirnov Statistic)**
- Compares the shape of two distributions (reference vs evaluation data)
- Asks: "How different do these two datasets look?"
- Returns a number between 0 (identical) and 1 (completely different)

**2. DFKI Gradient Weight**

This component measures how much the data's statistical properties changed:

- **Mean Gradient**: Measures how much the average value shifted
  - Example: If average review length was 100 words but became 150 words, that's a big shift
  - Calculated as: absolute difference in averages, normalized by the original spread

- **Variance Gradient**: Measures how much the data spread changed
  - Example: If reviews used to be consistently 80-120 words but now range from 20-300 words, the spread increased
  - Calculated as: absolute difference in spreads, normalized by the original spread

- **Combined Weight**: Takes both gradients and combines them to get overall change
  - Uses Pythagorean theorem to combine: square root of (mean_gradient² + variance_gradient²)

**3. Final Drift Score**

The final score multiplies the KS statistic by the gradient weight:
- **drift_score = KS_statistic × (1 + gradient_weight)**

This means:
- If distributions look different AND statistics changed a lot = HIGH drift score
- If distributions look similar AND statistics stayed stable = LOW drift score

**Score Interpretation:**
- **0.0 - 0.1**: Low drift - data is stable and reliable
- **0.1 - 0.3**: Medium drift - some changes occurred, monitor closely
- **≥0.3**: High drift - significant changes detected, model may need retraining


**Project Summary:** This experiment successfully demonstrates covariate drift detection using the DFKI gradient-based method. The metric identified significant drift in Amazon review features over time, which directly correlated with a 5.8% degradation in model performance.
