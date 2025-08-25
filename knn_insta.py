import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.neighbors as knn 

mydata=pd.read_csv("insta_post.csv")
x=mydata[["insta_posts"]] 
y=mydata[["followers_count"]] 

plt.scatter(x,y) 
plt.show()
model=knn.KNeighborsRegressor(n_neighbors=3)
model.fit(x,y)
print(model.predict([[100]]))

