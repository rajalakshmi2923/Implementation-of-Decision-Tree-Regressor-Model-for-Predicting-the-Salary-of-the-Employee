# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Moodle-Code Runner

## Algorithm
1. Import the required libraries.
2. Upload and read the dataset.
3. Gather information and presence of null in the dataset.
4. From sklearn.tree import DecisionTreeRegressor and fir the model.
5.Find the mean square error and r squared score value of the model.
6. Check the trained model.

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: R.Rajalakshmi
RegisterNumber: 212219040116
*/
import pandas as pd
data = pd.read_csv("Salary.csv")
data.head()
data.info()
data.isnull().sum()
from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()
x= data[["Position","Level"]]
y=data["Salary"]
from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.2,random_state = 2)
from sklearn.tree import DecisionTreeRegressor
dt = DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred = dt.predict(x_test)
from sklearn import metrics
mse = metrics.mean_squared_error(y_test,y_pred)
mse
r2= metrics.r2_score(y_test,y_pred)
r2
dt.predict([[5,6]])

```

## Output:
### Initial Dataset:
![Screenshot (162)](https://user-images.githubusercontent.com/87656716/173230536-7245e0ea-a021-4411-957c-7bee1e78dd1c.png)
### Dataset information:
![Screenshot (163)](https://user-images.githubusercontent.com/87656716/173230601-87c26093-c70a-4dad-ab83-792fb21cbb5e.png)
### Null dataset:
![Screenshot (164)](https://user-images.githubusercontent.com/87656716/173230704-2d0d9052-ed13-4427-8b71-eb211e232fce.png)
### Encoded Dataset:
![Screenshot (165)](https://user-images.githubusercontent.com/87656716/173230794-f1c456d8-134a-4180-a887-a93ad3fc1deb.png)
### Mean Square Error value:
![Screenshot (166)](https://user-images.githubusercontent.com/87656716/173230842-5c93b33e-f752-4132-b8ff-a933e255247c.png)
### R squared score:
![Screenshot (167)](https://user-images.githubusercontent.com/87656716/173230869-6e4bd2e6-b940-4301-b6e5-10ce651d6b9f.png)
### pedictted value:
![Screenshot (168)](https://user-images.githubusercontent.com/87656716/173230932-e7c8269c-1ceb-43aa-8f11-290102dd98d4.png)

## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
