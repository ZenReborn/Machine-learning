import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats

df=pd.read_csv('salary.csv')
X=df['YearsExperience']
y=df['Salary']

slope,intercept,r,p,std_error=stats.linregress(X,y)
print("r_value=", r)

def myfunc(x):
    return slope*x + intercept
plt.scatter(X,y,c='red')
plt.xlabel("Years of Experience")
plt.ylabel("Salary")
plt.plot(X,myfunc(X))
plt.show()
