import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("insta_post.csv")
x=mydata[["insta_posts"]] 
y=mydata[["followers_count"]] 
model=lm.LinearRegression()
model.fit(x,y)
plt.scatter(x,y) 
plt.plot(x,model.predict(x),color="red")
plt.show()
print(model.predict([[100]]))
print(model.coef_)
print(model.intercept_)

