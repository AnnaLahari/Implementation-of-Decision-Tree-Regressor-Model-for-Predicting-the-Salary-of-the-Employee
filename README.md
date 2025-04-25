# Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee

## AIM:
To write a program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the dataset Salary.csv using pandas and view the first few rows.

2.Check dataset information and identify any missing values.

3.Encode the categorical column "Position" into numerical values using LabelEncoder.

4.Define feature variables x as "Position" and "Level", and target variable y as "Salary".

5.Split the dataset into training and testing sets using an 80-20 split.

6.Create a DecisionTreeRegressor model instance.

7.Train the model using the training data.

8.Predict the salary values using the test data.

9.Evaluate the model using Mean Squared Error (MSE) and R² Score.

10.Use the trained model to predict salary for a new input [5, 6].

## Program:
```
/*
Program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee.
Developed by: A.LAHARI
RegisterNumber: 212223230111 
*/

import pandas as pd
data=pd.read_csv("Salary.csv")
data.head()

data.info

data.isnull().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data["Position"]=le.fit_transform(data["Position"])
data.head()

x=data[["Position","Level"]]
x.head()

y=data[["Salary"]]

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=2)

from sklearn.tree import DecisionTreeRegressor
dt=DecisionTreeRegressor()
dt.fit(x_train,y_train)
y_pred=dt.predict(x_test)

from sklearn import metrics
mse=metrics.mean_squared_error(y_test, y_pred)
mse

r2=metrics.r2_score(y_test,y_pred)
r2

dt.predict([[5,6]])

```

## Output:

### Data Head:
![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/b2f6f2eb-1e0c-4fbb-8784-4a8bd706c979)

### Data Info:

![image](https://github.com/user-attachments/assets/6ee5f82a-9ac8-4bd4-b6a9-098c8d56dfa7)

### isnull() sum():

![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/3a21fac0-df89-4aaf-827f-bc00aa3f0286)

### Data Head for salary:

![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/0a79abfa-f32d-4394-a73d-47161eaeec30)

### Mean Squared Error :

![image](https://github.com/user-attachments/assets/e50a8acc-4ae2-49f4-ba68-930a1bbf08a7)

![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/3c7acf12-adb7-4a3f-807e-cb49ad260032)

### r2 Value:

![image](https://github.com/HIRU-VIRU/Implementation-of-Decision-Tree-Regressor-Model-for-Predicting-the-Salary-of-the-Employee/assets/145972122/e6f5cab9-dab9-4c69-bb0e-6fa0abee1da0)

### Data prediction :

![image](https://github.com/user-attachments/assets/532c038e-30d3-4e4c-8d00-c0b02c3b3d9e)


## Result:
Thus the program to implement the Decision Tree Regressor Model for Predicting the Salary of the Employee is written and verified using python programming.
