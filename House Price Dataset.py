# 1. Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# --- Step 1 & 2: Load Data and Select/Engineer Features ---

# Load the training dataset using its full path
try:
    # Use the path to your train.csv file
    df = pd.read_csv(r'C:\Users\kartik rathod\Documents\House_Price_Project\train.csv')
except FileNotFoundError:
    print("Error: The file was not found. Please check the path to 'train.csv'.")
    exit()

# Engineer the 'TotalBath' feature
df['TotalBath'] = df['FullBath'] + (0.5 * df['HalfBath']) + df['BsmtFullBath'] + (0.5 * df['BsmtHalfBath'])

# --- THIS IS THE ONLY LINE THAT HAS CHANGED ---
# Define the features, now including TotalBsmtSF and GarageCars
features = ['GrLivArea', 'BedroomAbvGr', 'TotalBath', 'OverallQual', 'YearBuilt', 'TotalBsmtSF', 'GarageCars']
target = 'SalePrice'

# Create a new, clean DataFrame and handle potential missing values for the new features
house_df = df[features + [target]].dropna()


# --- Step 3: Define Features (X) and Target (y) ---
X = house_df[features]
y = house_df[target]


# --- Step 4: Split Data into Training and Testing Sets ---
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Training data has {X_train.shape[0]} samples.")
print(f"Testing data has {X_test.shape[0]} samples.")


# --- Step 5: Build and Train the Linear Regression Model ---
model = LinearRegression()
model.fit(X_train, y_train)
print("\nModel training complete! 🎉")


# --- Step 6: Test the Model and Evaluate Performance ---
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print("\n--- Model Performance ---")
print(f"Root Mean Squared Error (RMSE): ${rmse:,.2f}")
print(f"R-squared (R²): {r2:.2f}")


# --- Step 7: Interpret the Model's Logic ---
intercept = model.intercept_
coefficients = model.coef_

print("\n--- Model Interpretation ---")
print(f"Intercept (Base Price): ${intercept:,.2f}")
print("Coefficients (Price change per unit increase):")
for feature, coef in zip(features, coefficients):
    print(f"- {feature}: ${coef:,.2f}")