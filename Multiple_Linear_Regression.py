# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd
import numpy as np


#importing datasets
data_set= pd.read_csv('Advertising Budget and Sales.csv')
print(data_set.to_string())
print("Mean")
print(data_set.describe())

#Extracting Independent and dependent Variable
x= data_set.iloc[:, :-1].values
y= data_set.iloc[:, 4].values
df2=pd.DataFrame(x)
print("X=")
print(df2.to_string())
df3=pd.DataFrame(y)
print("Y=")
print(df3.to_string())

#Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test= train_test_split(x, y, test_size= 0.2, random_state=0)

#Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor= LinearRegression()
regressor.fit(x_train, y_train)

#Predicting the Test set result;
y_pred= regressor.predict(x_test)

#To compare the actual output values for X_test with the predicted value
df = pd.DataFrame({'Actual Data :-': y_test, 'Predicted Data :-': y_pred})
print(df.to_string())

#Evaluating the Algorithm
from sklearn import metrics
print('Mean Absolute Error     :', metrics.mean_absolute_error(y_test, y_pred))
print('Mean Squared Error      :', metrics.mean_squared_error(y_test, y_pred))
print('Root Mean Squared Error :', np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# predicting the accuracy score
from sklearn.metrics import r2_score
score=r2_score(y_test,y_pred)
print("r2 socre   :",score*100,"%")

k_test=[[8.0,48,9.0,75]]
y_p= regressor.predict(k_test)
print("Predicted Data :" ,y_p)