import pandas as pd
from matplotlib import pyplot as plt
import sklearn.linear_model as lm 
from sklearn.model_selection import train_test_split

mydata=pd.read_csv("youtube_video_length_vs_views_80k.csv")
x=mydata[["video_length_min"]]
y=mydata[["views_count"]]
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2)
model=lm.LinearRegression()
model.fit(x_train,y_train)
print(model.predict([[38.58]]))

print(model.coef_)
print(model.intercept_)
plt.scatter(x,y,color="green")
plt.xlabel("Video length in mins")
plt.ylabel("Views count")
plt.plot(x,model.predict(x),color="red")
plt.show()

