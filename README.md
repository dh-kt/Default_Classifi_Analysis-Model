# Default Classification - Credit Card Default Prediction

## Project Overview
Predict whether a customer will default on their credit card payment using balance and income.

**Author:** DhaBa  
**Date:** April 2025  
**Tools:** Python, scikit-learn, pandas, numpy

## Dataset

| Property | Value |
| Source | ISLP package (Default dataset) |
| Observations | 10,000 customers |
| Features | balance, income |
| Target | default (Yes/No) |

### Class Distribution
- No Default: 9,667 (96.7%)
- Default: 333 (3.3%)

**Key Insight:** Imbalanced data (only 3.3% default). Accuracy is misleading.

## Methods Implemented

| Method              | Type |

| Logistic Regression | Parametric |
| LDA                 | Parametric |
| QDA                 | Parametric |
| Naive Bayes         | Parametric |
| KNN (k=5)           | Non-parametric |

## Results at Default Threshold (0.5)

| Method             | Accuracy | AUC |
| Logistic Regression | 0.9695 | 0.9425 |
| LDA                 | 0.9680 | 0.9426 |
| QDA                 | 0.9695 | 0.9420 |
| Naive Bayes         | 0.9665 | 0.9400 |
| KNN (k=5)           | 0.9655 | 0.5000 |

**Finding:** KNN fails on imbalanced data (AUC = 0.50 = random guessing).

## Threshold Tuning Results (Logistic Regression)

| Threshold | Sensitivity | Precision | Predicted Yes |
| 0.1       | 0.62         | 0.08     | 256 |
| **0.2**   | **0.46**     | **0.36** | **88** |
| 0.3       | 0.38         | 0.44     | 60 |
| 0.4       | 0.30         | 0.52     | 40 |
| 0.5       | 0.25         | 0.60     | 28 |
| 0.6       | 0.18         | 0.68     | 18 |
| 0.7       | 0.12         | 0.75     | 11 |
| 0.8       | 0.07         | 0.80     | 6 |
| 0.9       | 0.03         | 0.85     | 2 |

## Final Recommendation

| Setting         | Value |
| **Model**       | Logistic Regression |
| **Threshold**   | 0.2 |
| **Sensitivity** | 46% (catches 46 out of 100 defaulters) |
| **Precision**   | 36% (1 in 3 flagged is correct) |
| **Predicted Yes** | 88 customers flagged as high-risk |

### Why Threshold 0.2?
- Missing a defaulter (False Negative) = HIGH cost
- False alarm (False Positive) = LOW cost
- Lower threshold catches more defaulters

### Business Impact
- Default threshold (0.5): catches 25 defaulters per 100
- Recommended threshold (0.2): catches 46 defaulters per 100
- Cost: 60 extra warning letters to prevent 21 more defaults

## Comparison at Threshold 0.2

| Method               | Sensitivity | Precision | Predicted Yes |
| Logistic Regression  | 0.46        | 0.36      | 88 |
| LDA                  | 0.46        | 0.40      | 80 |
| QDA                  | 0.52        | 0.37      | 97 |
| Naive Bayes          | 0.51        | 0.36      | 97 |
| KNN                  | 0.00        | 0.00       | 0 |

## Key Takeaways
1. **Accuracy is misleading** for imbalanced data (96.7% No, 3.3% Yes)
2. **Sensitivity and Precision** are better metrics
3. **Threshold tuning** improves business outcomes
4. **KNN fails** on imbalanced data (AUC = 0.50)
5. **Logistic Regression** is interpretable and performs well

## How to Run

```bash
# Clone repository
git clone https://github.com/dh-kt/boston-housing-analysis.git

# Install dependencies
pip install pandas numpy matplotlib scikit-learn ISLP

# Run the notebook
jupyter notebook default_classification_analysis.ipynb
