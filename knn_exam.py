import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.neighbors as knn 

mydata=pd.read_csv("study_Time.csv")
x=mydata[["study_time_hours"]] 
y=mydata[["exam_percentage"]] 

plt.scatter(x,y) 
plt.show()
model=knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[4.5]]))
