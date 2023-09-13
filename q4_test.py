# -*- coding: utf-8 -*-
"""Q4_test.ipynb
Automatically generated by Colaboratory.
Original file is located at
    https://colab.research.google.com/drive/1dUjXq9nEOX2PMOboEnFAB_-x-hQGM49x
"""
import numpy as np
import matplotlib.pyplot as plt
from preprocessing.polynomial_features import PolynomialFeatures
from linearRegression.linear_regression import LinearRegression
import pandas as pd
import os.path
from os import path
from metrics import *
np.random.seed(45)  #Setting seed for reproducibility
from sklearn.datasets import make_regression
X, y = make_regression(n_samples=100, n_features=2, noise=20, random_state=4)
if not path.exists('Plots/Question4/'):
  os.makedirs('Plots/Question4/')

# X = np.array([i*np.pi/180 for i in range(60,300,2)])
# y = 3*X + 8 + np.random.normal(0,3,len(X))
# X=X.reshape(-1,1)
#TODO : Write here
#Preprocess the input using the polynomial features
#Solve the resulting linear regression problem by calling any one of the 
#algorithms you have implemented.
y_axis = []

for degree in range(1,11):
  poly = PolynomialFeatures(degree=degree,include_bias=True)
  X_train = np.array([])
  for i in range(len(X)):
      if X_train.shape[0] != 0:
          X_train = np.vstack((X_train,poly.transform(X[i])))
      else:
          X_train = poly.transform(X[i])
  #print(X_train.shape,degree)
  LR = LinearRegression(fit_intercept=True)
  theta = LR.fit_sklearn_LR(X_train,y)
  y_hat = LR.predict(X_train)
  print(rmse(y_hat,y))
  #print(theta)
  # plt.scatter(X,y)
  # plt.plot(X,y_hat)
  y_axis.append(np.linalg.norm(theta))
y_axis = np.array(y_axis)
y_axis = y_axis/ np.linalg.norm(y_axis)
plt.plot([i for i in range(1,5)],y_axis[:4])
plt.title("Norm(θ) vs Degree")
plt.savefig('Plots/Question4/Norm(θ)_vs_Degree for 3.png')
plt.show()

plt.plot([i for i in range(1,11)],y_axis)
plt.title("Norm(θ) vs Degree")
plt.savefig('Plots/Question4/Question4_Norm(θ)_vs_Degree.png')
plt.show()
