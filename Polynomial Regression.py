# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

# importing datasets
data_set = pd.read_csv(r"Ice_cream selling data.csv")
print(data_set)

#Extracting Independent and dependent Variable
x= data_set.iloc[:,[0]].values
y= data_set.iloc[:,[1]].values

#convert to dataframe
df1=pd.DataFrame(x)
df2=pd.DataFrame(y)
print("level")
print(df1.to_string())
print("salary")
print(df2.to_string())

#Building the Linear regression model:
"""Now, we will build and fit the Linear regression model to the
dataset. In building polynomial regression,
we will take the Linear regression model as reference and compare
both the results. """

#Fitting the Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
lin_regs= LinearRegression()
lin_regs.fit(x,y)

#Building the Polynomial regression model:
"""Now we will build the Polynomial Regression model, but it will be
a little different from the Simple Linear model. Because here we will use PolynomialFeatures class of preprocessing
library. We are using this class to add
some extra features to our dataset."""

#Fitting the Polynomial regression to the dataset
from sklearn.preprocessing import PolynomialFeatures

#The parameter value(degree= 2) depends on our choice. We can choose it according to our Polynomial features.
poly_regs= PolynomialFeatures(degree= 2)

#we are converting our feature matrix into polynomial feature matrix, and then fitting it to the Polynomial regression
x_poly= poly_regs.fit_transform(x)
lin_reg_2 =LinearRegression()
lin_reg_2.fit(x_poly, y)

#print("done")
#Visualizing the result for Linear regression:
#Visulaizing the result for Linear Regression model
mtp.scatter(x,y,color="blue")
mtp.plot(x,lin_regs.predict(x), color="red")
mtp.title("Bluff detection model(Linear Regression)")
mtp.xlabel("Temperature (°C)")
mtp.ylabel("Ice Cream Sales (Units)")
mtp.show()

#Visualizing the result for Polynomial Regression
"""Here we will visualize the result of Polynomial regression model,
code
for which is little different from the above model."""

#Visulaizing the result for Polynomial Regression
mtp.scatter(x,y,color="blue")
mtp.plot(x, lin_reg_2.predict(poly_regs.fit_transform(x)), color="red")
mtp.title("Bluff detection model(Polynomial Regression)")
mtp.xlabel("Temperature (°C)")
mtp.ylabel("Ice Cream Sales (Units)")
mtp.show()

#Predicting the final result with the Linear Regression model:
"""Now, we will predict the final output using the Linear regression
model to see whether an employee is saying truth or bluff. So, for this,
we will use the predict() method and will pass the value 6.5. Below is
the code for it:"""

lin_pred = lin_regs.predict([[6.5]])
print(lin_pred)

#Predicting the final result with the Polynomial Regression model:
"""Now, we will predict the final output using the Polynomial
Regression model to compare
with Linear model. Below is the code for it:"""

poly_pred = lin_reg_2.predict(poly_regs.fit_transform([[6.5]]))
print(poly_pred)

