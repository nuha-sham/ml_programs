import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("slr.csv")
x=mydata[["exp"]] 
y=mydata[["per"]] 
plt.scatter(x,y) 
plt.show()
model=lm.LinearRegression()
model.fit(x,y)
print(model.predict([[30]]))
print(model.coef_)
print(model.intercept_)