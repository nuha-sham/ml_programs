import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("height_weight_data.csv")
x=mydata[["height"]] 
y=mydata[["weight"]] 
plt.scatter(x,y) 
plt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[160]]))
print(model.coef_)
print(model.intercept_)