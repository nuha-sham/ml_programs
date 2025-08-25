import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
import sklearn.neighbors as knn 
mydata=pd.read_csv("data_new.csv")
x=mydata[["Age","Height"]] 
y=mydata[["Weight"]] 
model=lm.LinearRegression()
model1=knn.KNeighborsRegressor(n_neighbors=3)
model1.fit(x,y)
model.fit(x,y)
print(model.predict([[32,160]]))
print(model1.predict([[32,160]]))

