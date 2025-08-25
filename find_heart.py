import pandas as pd
import sklearn.linear_model as lm 
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import math
mydata=pd.read_csv("heart.csv")
x=mydata[["age","sex","cp","trestbps","chol","fbs","restecg","thalach","exang","oldpeak","slope","ca","thal"]]
y=mydata[["target"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=lm.LinearRegression()
model.fit(x_train,y_train)
print(model.predict([[71,0,0,112,149,0,1,125,0,1.6,1,0,2]]))

y_prediction_value=model.predict(x_test)
mse=mean_squared_error(y_test,y_prediction_value)
print("MSE",mse)
print("RMSE",math.sqrt(mse))
