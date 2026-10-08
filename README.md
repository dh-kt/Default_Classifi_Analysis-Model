# Default Classification - Credit Card Default Prediction

## Project Overview

This project develops and evaluates classification models for predicting credit card default risk using the ISLP Default dataset.

**Author:** DhaBa  
**Date:** Sept 2026  
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

**Key Insight:** Only 3.3% of customers default. This class imbalance means that accuracy alone is not sufficient for model evaluation. Precision, recall, and ROC-AUC are therefore considered throughout the analysis.

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

**Finding:** While KNN achieves high accuracy, its ROC-AUC score of 0.50 indicates no ability to distinguish between defaulting and non-defaulting customers. This demonstrates the limitations of relying solely on accuracy in highly imbalanced datasets.

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

### Threshold Tuning Findings

Lowering the classification threshold increases recall by identifying a larger proportion of actual defaulters, but it also increases the number of false positives.

Conversely, higher thresholds improve precision by generating fewer false alarms, but they miss a greater number of default cases.

Among the evaluated thresholds, 0.2 and 0.3 provide the most practical balance between sensitivity and precision. Threshold 0.2 identifies the highest proportion of defaulting customers, while threshold 0.3 achieves the strongest balance between recall and precision.

Given the objective of minimizing missed default cases, threshold 0.2 is selected for further evaluation and business recommendation.

## Final Recommendation

| Setting         | Value |

| **Model**       | Logistic Regression |

| **Threshold**   | 0.2 |

| **Sensitivity** | 46% (catches 46 out of 100 defaulters) |

| **Precision**   | 36% (1 in 3 flagged is correct) |

| **Predicted Yes** | 88 customers flagged as high-risk |

### Why Threshold 0.2?

- Missing a defaulter has a higher business cost than reviewing a low-risk customer.
- A lower threshold identifies substantially more high-risk customers.
- Logistic Regression provides interpretable probability estimates suitable for risk scoring.

### Business Impact
- Default threshold (0.5): catches 25 defaulters per 100
- Recommended threshold (0.2): catches 46 defaulters per 100
- Cost: 60 extra warning letters to prevent 21 more defaults

## Comparison at Threshold 0.2

| Method                 | Sensitivity | Precision | Predicted Yes |

| Logistic Regression    | 0.46        | 0.36      | 88 |

| LDA                    | 0.46        | 0.40      | 80 |

| QDA                    | 0.52        | 0.37      | 97 |

| Naive Bayes            | 0.51        | 0.36      | 97 |

| KNN                    | 0.00        | 0.00       | 0 |

## Key Takeaways
1. **Accuracy is misleading** for imbalanced data (96.7% No, 3.3% Yes)
2. **Sensitivity and Precision** are better metrics
3. **Threshold tuning** improves business outcomes
4. **KNN fails** on imbalanced data (AUC = 0.50)
5. **Logistic Regression** provides the best balance of interpretability, discrimination performance, and practical business value. 

## How to Run

```bash
# Clone repository

git clone https://github.com/dh-kt/credit_card-default-prediction.git

# Install dependencies
pip install pandas numpy matplotlib scikit-learn ISLP

# Run the notebook
jupyter notebook default_classification_analysis.ipynb
