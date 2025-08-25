import pandas as pd
from matplotlib import pyplot as plt
import sklearn.linear_model as lm 
from sklearn.model_selection import train_test_split

mydata=pd.read_csv("weather_data_80k.csv")
x=mydata[["Humidity(%)","WindSpeed","Rainfall(mm)","AirPressure"]]
y=mydata[["Temperature"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=lm.LinearRegression()
model.fit(x_train,y_train)
print(model.predict([[65,13.2,2.3,1008.9]]))


