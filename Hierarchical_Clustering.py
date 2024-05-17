# Importing the libraries
import numpy as nm
import matplotlib.pyplot as mtp
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Salary_Data.csv')
dataset.dropna(inplace=True)

#Form a dataframe
df=pd.DataFrame(dataset)
print(df.to_string())

#Extracting the matrix of features
x = dataset.iloc[:, [4,5]].values  #Annual income and spending score

#Finding the optimal number of clusters using the Dendrogram
import scipy.cluster.hierarchy as shc
dendro = shc.dendrogram(shc.linkage(x, method="ward"))
mtp.title("Dendrogrma Plot")
mtp.ylabel("Euclidean Distances")
mtp.xlabel("Customers")
mtp.show()

"""In the above lines of code, we have imported the
hierarchy module of scipy library.
This module provides us a method shc.denrogram(),
which takes the linkage() as a parameter.
The linkage function is used to define the distance
between two clusters, so here we have
passed the x(matrix of features), and method "ward,"
the popular method of linkage in
hierarchical clustering."""

#training the hierarchical model on dataset
from sklearn.cluster import AgglomerativeClustering
hc= AgglomerativeClustering(n_clusters=5, affinity='euclidean', linkage='ward')
y_pred= hc.fit_predict(x)
print(y_pred)