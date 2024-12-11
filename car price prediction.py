# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.metrics import r2_score
import joblib

# 1. Load the Dataset
car_dataset = pd.read_csv(r"C:\Users\KIIT\Downloads\archive (2)\car data.csv")
print("First 5 rows of the dataset:")
print(car_dataset.head())

# Inspect dataset
print("\nDataset Info:")
car_dataset.info()

print("\nNumber of missing values in each column:")
print(car_dataset.isnull().sum())

# Check distribution of categorical variables
print("\nCategorical Data Distribution:")
print("Fuel_Type:\n", car_dataset.Fuel_Type.value_counts())
print("Seller_Type:\n", car_dataset.Seller_Type.value_counts())
print("Transmission:\n", car_dataset.Transmission.value_counts())

# 2. Preprocess the Data
car_dataset.replace({'Fuel_Type': {'Petrol': 0, 'Diesel': 1, 'CNG': 2}}, inplace=True)
car_dataset.replace({'Seller_Type': {'Dealer': 0, 'Individual': 1}}, inplace=True)
car_dataset.replace({'Transmission': {'Manual': 0, 'Automatic': 1}}, inplace=True)

# Define features (X) and target (Y)
X = car_dataset.drop(['Car_Name', 'Selling_Price'], axis=1)
Y = car_dataset['Selling_Price']

print("\nFeatures (X):")
print(X.head())
print("\nTarget (Y):")
print(Y.head())

# 3. Split the Data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.1, random_state=2)

# 4. Train Linear Regression Model
lin_reg_model = LinearRegression()
lin_reg_model.fit(X_train, Y_train)

# Save the model
joblib.dump(lin_reg_model, 'linear_reg_model.pkl')
print("\nLinear Regression model saved as 'linear_reg_model.pkl'.")

# Evaluate on Training Data
training_predictions = lin_reg_model.predict(X_train)
train_r2 = r2_score(Y_train, training_predictions)
print("\nLinear Regression R-squared (Training Data):", train_r2)

# Evaluate on Test Data
test_predictions = lin_reg_model.predict(X_test)
test_r2 = r2_score(Y_test, test_predictions)
print("Linear Regression R-squared (Test Data):", test_r2)

# Plot Actual vs Predicted (Training Data)
plt.figure(figsize=(6, 6))
plt.scatter(Y_train, training_predictions, alpha=0.7, color='blue')
plt.title("Linear Regression: Actual vs Predicted (Training Data)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.show()

# Plot Actual vs Predicted (Test Data)
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, test_predictions, alpha=0.7, color='green')
plt.title("Linear Regression: Actual vs Predicted (Test Data)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.show()

# 5. Train Lasso Regression Model
lasso_reg_model = Lasso()
lasso_reg_model.fit(X_train, Y_train)

# Evaluate on Training Data
training_predictions_lasso = lasso_reg_model.predict(X_train)
train_r2_lasso = r2_score(Y_train, training_predictions_lasso)
print("\nLasso Regression R-squared (Training Data):", train_r2_lasso)

# Evaluate on Test Data
test_predictions_lasso = lasso_reg_model.predict(X_test)
test_r2_lasso = r2_score(Y_test, test_predictions_lasso)
print("Lasso Regression R-squared (Test Data):", test_r2_lasso)

# Plot Actual vs Predicted (Lasso, Training Data)
plt.figure(figsize=(6, 6))
plt.scatter(Y_train, training_predictions_lasso, alpha=0.7, color='red')
plt.title("Lasso Regression: Actual vs Predicted (Training Data)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.show()

# Plot Actual vs Predicted (Lasso, Test Data)
plt.figure(figsize=(6, 6))
plt.scatter(Y_test, test_predictions_lasso, alpha=0.7, color='purple')
plt.title("Lasso Regression: Actual vs Predicted (Test Data)")
plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")
plt.show()
