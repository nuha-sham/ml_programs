import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.neighbors as knn 

mydata=pd.read_csv("plants.csv")
x=mydata[["days_watered"]] 
y=mydata[["plant_height_cm"]] 

plt.scatter(x,y) 
plt.show()
model=knn.KNeighborsRegressor(n_neighbors=4)
model.fit(x,y)
print(model.predict([[7.5]]))
