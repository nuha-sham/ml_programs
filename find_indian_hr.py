import pandas as pd
import sklearn.linear_model as lm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
mydata=pd.read_csv("indian_house_rent_dataset.csv")
x=mydata[["CityTier","DistanceFromCityCenter","HouseSize","NoOfRooms","AgeOfBuilding"]]
y=mydata[["Rent_INR"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=lm.LinearRegression()
model.fit(x_train,y_train)
print(model.predict([[2,27.56,2043,2,29]]))

y_prediction_value=model.predict(x_test)
mse=mean_squared_error(y_test,y_prediction_value)
print("MSE",mse)
print("RMSE",math.sqrt(mse))
