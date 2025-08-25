import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("study_Time.csv")
x=mydata[["study_time_hours"]] 
y=mydata[["exam_percentage"]] 
model=lm.LinearRegression()
model.fit(x,y)
plt.scatter(x,y) 
plt.plot(x,model.predict(x),color="red")
plt.show()
print(model.predict([[4.5]]))
print(model.coef_)
print(model.intercept_)
