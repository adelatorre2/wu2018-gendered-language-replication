# Methodology Notes (Extension)

## Why these models?
- **OLS / Linear Probability Model (LPM)** is a simple, interpretable baseline. Coefficients map directly to word associations, making it easy to compare with Wu (2018) Table 1. The tradeoff is that predicted values are not bounded in [0, 1].
- **Random Forest (RF)** captures nonlinearities and interactions that LPM cannot. It provides a complementary, flexible model but is heavier to train and harder to interpret.

## Reuse of upstream logic
- Labels and train/test splits are **exactly** those defined in the upstream Lasso scripts:
  - training == 1 is the training set (non-duplicate posts)
  - training == 0 is the test set
  - posts with both male and female classifiers (training is NA) are excluded
- Features are restricted using the same `i_keep_columns` logic (exclude gender classifiers and other banned terms).

## Feature reduction for Random Forest
- The original feature space has ~9,540 predictors after exclusions.
- Random Forests are inefficient with large sparse text matrices. To keep this extension feasible:
  - We keep **top-k features by document frequency** on the training set.
  - We **subsample rows** for training and test evaluation.
- This is documented in the output metrics and logs. It is a *pragmatic* adjustment, not a change in the core data definition.

## Directional interpretation for RF
- RF feature importances are unsigned. To recover a “female vs male” direction:
  - Compute sign(mean(feature|female=1) − mean(feature|female=0)).
  - Combine the sign with RF importance to list top words for each direction.
- This is a heuristic, not a causal statement.

## Limitations
- **Prediction vs. causation:** These models identify words associated with gender-referenced posts; they do not imply causality.
- **Label noise:** Gender labels come from text classifiers and may include errors or ambiguous cases.
- **Representativeness:** EJMR is an anonymous forum; language may not reflect broader academic culture.
- **Model tradeoffs:** LPM is linear and unbounded; RF is nonlinear but less interpretable and sensitive to feature frequency.

## What this extension can and cannot say
- **Can:** Compare alternative predictive models on the same data/split and see whether the most predictive words broadly align with Wu (2018).
- **Cannot:** Establish causal claims about language or generalize beyond the EJMR context.
