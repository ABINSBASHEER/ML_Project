# importing libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

# importing datasets
data_set = pd.read_csv('Advertising Budget and Sales.csv')
print(data_set.to_string())

# Extracting Independent and dependent Variable
x = data_set.iloc[:, :-1].values
y = data_set.iloc[:, 4].values
# display x and y
print("X=",x)
print("Y=",y)

# Splitting the dataset into training and test set.
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

# Fitting the MLR model to the training set:
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(x_train, y_train)

# Predicting the Test set result;
y_pred = regressor.predict(x_test)
df6=pd.DataFrame({'Actual Data  :':y_test,'Predicted Data :':y_pred})
print(df6)

# Checking the score
print('Train Score: ', regressor.score(x_train, y_train)*100,'%')
print('Test Score: ', regressor.score(x_test,y_test)*100,'%')

pre=[[62.3,12.6,18.3,9.7]]
pred=regressor.predict(pre)
print(pred)