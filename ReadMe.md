# S&P 500 Price Prediction
This project builds a machine learning pipeline to predict the next-day direction (Up/Down) of the S&P 500 index using historical market data and engineered technical indicators.

# Project Overview
Historical daily data for the S&P 500 (^GSPC) is downloaded using yfinance.
Data from 1990 onward is used for modeling.
The target variable is binary:
1 = Next day price goes up
0 = Next day price goes down

### Feature Engineering
A comprehensive set of technical and statistical features was created:
Daily returns and lagged returns (momentum signals)
Rolling mean returns (trend strength)
Rolling volatility (market regime detection)
Price-to-moving-average ratios (10, 20, 50, 200-day)
Intraday range & gap features
Volume-based features
RSI (14-day Relative Strength Index)
Day-of-week encoding

### Models Implemented:
1-Logistic Regression
2-Support Vector Machine (RBF kernel)
3-Random Forest

### Model Training Strategy:
Time-series split (no data leakage)
GridSearchCV for hyperparameter tuning
Class balancing (upsampling & class weights)
StandardScaler applied where needed
Evaluation on a strict 80% train / 20% test chronological split

### Evaluation Metrics
Models were evaluated using:
Accuracy
Precision
Recall
F1-score
ROC-AUC
Balanced Accuracy
Matthews Correlation Coefficient (MCC)
Cohen’s Kappa
Confusion Matrix
ROC Curves

### Why is the Accuracy ~51%?
The model achieved an accuracy of approximately 51%, which may seem low at first glance. However, in financial time-series prediction, this is not necessarily poor performance.
Stock market movements are highly noisy, nonlinear, and close to random in the short term. Since the target is binary (Up/Down), a random guess would produce around 50% accuracy. Therefore, achieving 51% suggests the model is capturing a small but real predictive signal above randomness.
##### In financial markets:
Even a 1–2% edge can be valuable.
Small predictive advantages can be profitable when combined with proper risk management.
Market efficiency limits how predictable next-day movements can be.
##### Additionally:
The dataset spans decades of changing market regimes.
Daily returns are extremely volatile and influenced by external macroeconomic and geopolitical factors not included in the model.
The problem is inherently difficult due to weak signal-to-noise ratio.
Therefore, while 51% may appear modest, in the context of short-term market direction prediction, it represents a meaningful learning outcome and demonstrates how challenging financial forecasting truly is.
