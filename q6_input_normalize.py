from sklearn.datasets import load_diabetes
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
# load the diabetes dataset
n_samples = 200
split = 0.8
split_ind = int(n_samples*split)
X, y = make_regression(n_samples=n_samples, n_features=5, random_state=52)
#Preprocess the input using the polynomial features
X_train, y_train = X[:split_ind], y[:split_ind]
X_test, y_test = X[split_ind:], y[split_ind:]

# split the data into training and testing sets

# create the scaler
scaler = StandardScaler()

# fit and transform the training data with the scaler
X_train_scaled = scaler.fit_transform(X_train)

# transform the testing data with the scaler
X_test_scaled = scaler.transform(X_test)

# create the linear regression model
model = LinearRegression()

# fit the model on the unnormalized data
model.fit(X_train, y_train)

# predict on the unnormalized testing data and calculate the error
y_pred = model.predict(X_test)
error_unnormalized = np.mean((y_pred-y_test)**2)
print("No norm MSE : ",error_unnormalized)

# fit the model on the normalized data
model.fit(X_train_scaled, y_train)

# predict on the normalized testing data and calculate the error
y_pred_scaled = model.predict(X_test_scaled)
error_normalized = np.mean((y_test - y_pred_scaled)**2)
print("Norm MSE : ",error_normalized)

# plot the errors with both normalized and unnormalized data
plt.bar(['Unnormalized', 'Normalized'], [error_unnormalized, error_normalized])
plt.ylabel('Mean Squared Error')
plt.savefig('./Plots/Question6/q6 bar.png')
error_unnorm_plot = y_test - y_pred
error_norm_plot = y_test - y_pred_scaled
plt.scatter(y_pred, error_unnorm_plot, label='Unnormalized')
plt.scatter(y_pred_scaled, error_norm_plot, label='Normalized')
plt.legend()
plt.xlabel('Predicted Values')
plt.ylabel('Errors')
plt.title('Errors for Unnormalized and Normalized Data')
plt.savefig('./Plots/Question6/q6 scatter.png')

