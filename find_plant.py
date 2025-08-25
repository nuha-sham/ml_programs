import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("plants.csv")
x=mydata[["days_watered"]] 
y=mydata[["plant_height_cm"]] 
model=lm.LinearRegression()
model.fit(x,y)
plt.scatter(x,y) 
plt.plot(x,model.predict(x),color="red")
plt.show()
print(model.predict([[7.5]]))
print(model.coef_)
print(model.intercept_)
