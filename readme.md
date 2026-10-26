# Real Estate Housing Price Prediction (Ridge & Lasso Regression)

Predicts median house values using the California Housing dataset and compares
OLS, Ridge (L2) and Lasso (L1) regression models.

## What this project does
- Imputes missing values (`total_bedrooms`) using the median
- Engineers interaction features (rooms per household, income x rooms, etc.)
- Checks multicollinearity with a correlation heatmap and VIF
- Standardizes numeric features and one-hot encodes `ocean_proximity`
- Tunes alpha for Ridge and Lasso using GridSearchCV
- Evaluates models using RMSE, MAE and R²
- Plots predicted vs actual values and residual distributions

## Results (test set)
| Model | RMSE | MAE | R² |
|-------|------|-----|----|
| OLS   | 68,909 | 49,579 | 0.638 |
| Ridge | 68,910 | 49,580 | 0.638 |
| Lasso | 69,026 | 49,574 | 0.636 |

All three models perform almost the same because the dataset is large
relative to the number of features. Ridge stabilizes collinear coefficients,
and Lasso zeroes out weak features. Residuals fan out at higher prices, and
house values are capped at 500k, which shows the limit of a linear model.

## Files
- `housing_regression.ipynb` - main notebook with code, plots and conclusion
- `housing.py` - same code as a Python script
- `housing.txt` - dataset (CSV format)
- 

## How to run