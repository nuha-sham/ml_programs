import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.neighbors as knn 

mydata=pd.read_csv("screen.csv")
x=mydata[["screen_time"]] 
y=mydata[["eye_strain_score"]] 

plt.scatter(x,y) 
plt.show()
model=knn.KNeighborsRegressor(n_neighbors=5)
model.fit(x,y)
print(model.predict([[4.5]]))

