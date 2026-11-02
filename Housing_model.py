#you will implement linear regression with multiple variables to
#predict the prices of houses. Suppose you are selling your house and you
#want to know what a good market price would be. One way to do this is to
#first collect information on recent houses sold and make a model of housing
#prices.
#The file Housing_data.txt contains a training set of housing prices in Portland, Oregon. The first column is the size of the house (in square feet), the
#second column is the number of bedrooms, and the third column is the price of the house.
#
#  Instructions
#  ------------
# 
#  This file will give you outlines to get started on buiding a linear regression model. 
#

## ================ Part 1: Feature Normalization ================

## Load Housing_data.txt
# ====================== YOUR CODE HERE ======================
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd



####################################################################

# Print out some data points
# ====================== YOUR CODE HERE ======================

data_file = Path(__file__).resolve().with_name("Housing_data (2).txt")
data = pd.read_csv(
	data_file,
	header=None,
	names=["size", "bedrooms", "price"],
)
X_raw = data[["size", "bedrooms"]].to_numpy()
y = data["price"].to_numpy()
m = y.size


print("First 10 training examples (size, bedrooms, price):")
print(data.head(10))
####################################################################


# Scale features and set them to zero mean
# ====================== YOUR CODE HERE ======================
#               First, for each feature dimension, compute the mean
#               of the feature and subtract it from the dataset.  
#               Next, compute the standard deviation of each feature and divide
#               Next, You need to perform the normalization separately for each feature. 


mu = np.mean(X_raw, axis=0)
sigma = np.std(X_raw, axis=0)
X_norm = (X_raw - mu) / sigma
X = np.column_stack((np.ones(m), X_norm))


####################################################################

# Print out some data points
# ====================== YOUR CODE HERE ======================
print("\nFirst 10 normalized examples:")
print(X[:10, 1:])
print("Feature means:", mu)
print("Feature standard deviations:", sigma)


####################################################################

## =================== Part 2: Cost and Gradient descent ===================
# ====================== YOUR CODE HERE ======================
def compute_cost(features, prices, parameters):
	errors = features @ parameters - prices
	return (errors @ errors) / (2 * prices.size)


# ====================== YOUR CODE HERE ======================
# Some gradient descent settings
iterations = 1500
alpha = 0.01
iterations = 1500;
alpha = 0.01;
theta = np.zeros(X.shape[1])
J_history = np.zeros(iterations)

# Init Theta and Run Gradient Descent 

for iter in range(iterations):
	# Perform a single gradient step on the parameter thetas.
	errors = X @ theta - y
	theta -= alpha * (X.T @ errors) / m

    
	
#   After updating theta, compute the cost of using the current values of theta to fit the data points 
#   You should set J_history(iter) to the cost of using the current values of theta. 
	J_history[iter] = compute_cost(X, y, theta)
#   Save the cost J in every iteration    
    
    
####################################################################

plt.plot(np.arange(iterations), J_history)
plt.xlabel("Iteration")
plt.ylabel("Cost")
plt.title("Gradient Descent Convergence")
plt.tight_layout()
plt.show()
# Plot the convergence graph
# ====================== YOUR CODE HERE ======================


####################################################################

print("\nFinal parameters (intercept, normalized feature coefficients):")
print(theta)
print(f"Final training cost: {J_history[-1]:.2f}")
# Display gradient descent's result
# ====================== YOUR CODE HERE ======================


####################################################################

## Estimathe the price of a 1650 sq-ft, 3 br house
house = np.array([1650, 3])
house_norm = (house - mu) / sigma
house_features = np.concatenate(([1.0], house_norm))
predicted_price = house_features @ theta
print(f"Estimated price for a 1650 sq-ft, 3-bedroom house: ${predicted_price:,.2f}")
# ====================== YOUR CODE HERE ======================



####################################################################
