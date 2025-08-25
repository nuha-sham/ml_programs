import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("temperature_vs_sales.csv")
x=mydata[["temperature_C"]] 
y=mydata[["ice_cream_sales_$"]] 
model=lm.LinearRegression()
model.fit(x,y)
plt.scatter(x,y) 
plt.plot(x,model.predict(x),color="red")
plt.show()
print(model.predict([[35.23]]))
print(model.coef_)
print(model.intercept_)
