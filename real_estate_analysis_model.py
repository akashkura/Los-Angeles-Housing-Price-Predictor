# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load and compile dataset
# Assuming the dataset is stored in a CSV file named "housing_data.csv"
# Columns: "price", "sqft", "bedrooms", "age", "location"
data = pd.read_csv("housing_data.csv")

# Step 2: Data Cleaning and Preprocessing
# Handle missing values - typically by filling or dropping based on context
data.fillna(data.mean(), inplace=True)

# Step 3: Outlier Detection and Removal
# Here, we filter out outliers based on the interquartile range (IQR) for numerical features
for column in ["price", "sqft", "bedrooms", "age"]:
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    data = data[(data[column] >= (Q1 - 1.5 * IQR)) & (data[column] <= (Q3 + 1.5 * IQR))]

# Step 4: Correlation Analysis
# Check the correlation between home size (sqft) and price
correlation = data["sqft"].corr(data["price"])
print(f"Correlation between home size and price: {correlation:.2f}")  # Expected: ~0.25

# Step 5: Prepare Data for Model
# Define features (X) and target (y)
X = data[["sqft", "bedrooms", "age"]]
y = data["price"]

# Split the dataset into training and test sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train a Linear Regression Model with Regularization
# Initialize models with regularization
lr_model = LinearRegression()
lasso_model = Lasso(alpha=0.1)  # Lasso regularization with alpha value
ridge_model = Ridge(alpha=1.0)  # Ridge regularization with alpha value

# Train each model
lr_model.fit(X_train, y_train)
lasso_model.fit(X_train, y_train)
ridge_model.fit(X_train, y_train)

# Step 7: Model Evaluation
# Predict on test set and calculate accuracy (R^2 score) for each model
lr_pred = lr_model.predict(X_test)
lasso_pred = lasso_model.predict(X_test)
ridge_pred = ridge_model.predict(X_test)

# Calculate R^2 scores
lr_accuracy = r2_score(y_test, lr_pred)
lasso_accuracy = r2_score(y_test, lasso_pred)
ridge_accuracy = r2_score(y_test, ridge_pred)

print(f"Linear Regression Accuracy: {lr_accuracy:.2f}")
print(f"Lasso Regression Accuracy: {lasso_accuracy:.2f}")
print(f"Ridge Regression Accuracy: {ridge_accuracy:.2f}")

# Step 8: Visualize Trends
# Using seaborn and matplotlib for easy-to-understand visualizations
plt.figure(figsize=(10, 6))
sns.regplot(x="sqft", y="price", data=data)
plt.title("Relationship between Home Size (sqft) and Price")
plt.xlabel("Square Footage")
plt.ylabel("Price")
plt.show()

# Step 9: Provide ML-based recommendations based on trends
# Displaying the coefficient values to understand impact of each feature on pricing
print("Feature Importance in Linear Regression:")
for feature, coef in zip(X.columns, lr_model.coef_):
    print(f"{feature}: {coef:.2f}")

