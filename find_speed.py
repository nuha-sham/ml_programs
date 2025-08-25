import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("speed.csv")
x=mydata[["speed"]] 
y=mydata[["fuel_efficiency"]] 
model=lm.LinearRegression()
model.fit(x,y)
plt.scatter(x,y) 
plt.plot(x,model.predict(x),color="red")
plt.show()
print(model.predict([[65]]))
print(model.coef_)
print(model.intercept_)
