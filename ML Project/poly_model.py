import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score


df=pd.read_csv('ice_cream.csv')
X=df['Temperature']
y=df['Ice Cream Sales']

mymodel=np.poly1d(np.polyfit(X,y,3))
myline=np.linspace(X[0],X[len(X)-1],100)

poly=PolynomialFeatures(degree=2)
X_poly=poly.fit_transform(X.values.reshape(-1,1))
model=LinearRegression()
model.fit(X_poly,y)

y_pred=model.predict(X_poly)
r2=r2_score(y,y_pred)
print("r2_score= ",r2)

plt.xlabel("Temperature")
plt.ylabel("No of units sold")
plt.scatter(X,y,color='hotpink')
plt.plot(myline,mymodel(myline))
plt.show()
