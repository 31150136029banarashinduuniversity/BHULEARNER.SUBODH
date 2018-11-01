# Polynomial Linear Regression

#  importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# importing the datasets
dataset=pd.read_csv('Position_Salaries.csv')
x=dataset.iloc[:,1:2].values
y=dataset.iloc[:,2].values

# Fitting Linear Regression to the datasets
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(x,y)

# Fitting ploynomial linear regression to the datasets with degree2
"""from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=2)
x_poly=poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)"""


# Fitting ploynomial linear regression to the datasets with degree  3
"""from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=3)
x_poly=poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)"""

# Fitting ploynomial linear regression to the datasets with degree  4
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree=4)
x_poly=poly_reg.fit_transform(x)
lin_reg_2 = LinearRegression()
lin_reg_2.fit(x_poly,y)

# Visualising the Linear Regression Result
plt.scatter(x,y,color='red', s=100)
plt.plot(x,lin_reg.predict(x),color='blue',linewidth=10)
plt.title('Truth or Bluff(Linear Regression)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial Linear Regression Results
plt.scatter(x,y,color='red', s=10)
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue',linewidth=.5)
plt.title('Truth or Bluff(polynomial fitting)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.show()

# Visualising the polynomial Linear Regression Results
"""plt.scatter(x,y,color='red', s=10)
plt.plot(x,lin_reg_2.predict(poly_reg.fit_transform(x)),color='blue',linewidth=3)
plt.title('Truth or Bluff(polynomial fitting)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.ion()
plt.show()"""


# Visualising the polynomial Linear Regression Results
"""x_grid=np.arange(min(x),max(x),0.1)
x_grid=x_grid.reshape((len(x_grid),1))
plt.scatter(x,y,color='red', s=10)
plt.plot(x_grid,lin_reg_2.predict(poly_reg.fit_transform(x_grid)),color='blue',linewidth=3)
plt.title('Truth or Bluff(polynomial fitting)')
plt.xlabel('Position level')
plt.ylabel('Salary')
plt.ion()
plt.show()"""

# Predicting new result with linear regression
lin_reg.predict(6.5)

# predicting new result with Polynomial Regression
lin_reg_2.predict(poly_reg.fit_transform(6.5))