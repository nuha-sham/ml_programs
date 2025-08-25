import pandas as pd
from matplotlib import pyplot as plt 
import sklearn.linear_model as lm
mydata=pd.read_csv("weather_data_80k.csv")
x=mydata[["Humidity(%)","WindSpeed","Rainfall(mm)","AirPressure"]] 
y=mydata[["Temperature"]] 
model=lm.LinearRegression()

model.fit(x,y)
print(model.predict([[35.23]]))
