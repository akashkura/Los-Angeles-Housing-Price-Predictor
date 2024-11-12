# Los-Angeles-Housing-Price-Predictor
Project Description
This repository contains a predictive model that estimates housing prices in the Westwood, Los Angeles area. Using a dataset of homes priced between $900,000 and $1 million, the model employs Lasso and Ridge regularization techniques to provide accurate price predictions.

Table of Contents
Overview
Dataset
Methodology
Modeling
Lasso Regularization
Ridge Regularization
Installation
Usage
Results
Contributing
License
Overview
The Los Angeles Housing Price Predictor project is designed to predict home prices in the Westwood area of Los Angeles, focusing on houses within the $900,000 to $1 million price range. By utilizing both Lasso and Ridge regularization techniques, the model aims to mitigate overfitting and deliver robust predictions.

Key Features
Predict housing prices based on location-specific features.
Reduce model complexity and prevent overfitting using Lasso and Ridge regularization.
Optimize hyperparameters using grid search for improved performance.
Dataset
The dataset primarily includes homes in the 90024 Westwood area and consists of various features, including:

Number of bedrooms
Number of bathrooms
Square footage
Lot size
Year built
Proximity to amenities (schools, shopping centers, etc.)
Neighborhood quality
This dataset was self-collected or curated, and cleaned to focus on properties in the specified price range. The data preprocessing steps handle missing values, outliers, and feature scaling to improve model performance.

Methodology
The project follows a typical machine learning pipeline:

Data Collection & Cleaning: Outliers, missing values, and irrelevant features are addressed.
Feature Engineering: Creation of relevant features such as proximity to amenities, neighborhood quality, and property-specific features.
Model Selection: Both Lasso and Ridge regularization techniques are applied to prevent overfitting.
Hyperparameter Tuning: Grid search is used to optimize the regularization parameters (alpha for Lasso/Ridge).
Modeling
Lasso Regularization
Lasso (Least Absolute Shrinkage and Selection Operator) is used to:

Reduce coefficients of less significant features to zero, effectively performing feature selection.
Prevent overfitting by penalizing large coefficients.
Ridge Regularization
Ridge regression applies an L2 penalty to:

Regularize large coefficients, but unlike Lasso, it does not shrink coefficients to zero.
Improve model stability when features are highly correlated.
Both methods are tested and evaluated to determine which provides the best predictive performance for housing prices in the dataset.


Commit your changes (git commit -m 'Add some feature').
Push to the branch (git push origin feature/your-feature).
Open a pull request.
License
This project is licensed under the MIT License. See the LICENSE file for more details.
