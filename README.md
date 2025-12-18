# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values
4. Using logistic regression find the predicted values of accuracy , confusion matrices
5. Display the results.

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: DEEPAK KUMAR V
RegisterNumber: 212224240029
*/
```
```
import pandas as pd
data=pd.read_csv('Placement_Data.csv')
data.head()

data1=data.copy()
data1=data1.drop(["sl_no","salary"],axis=1)#Browses the specified row or column
data1.head()

data1.isnull().sum()

data1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
data1["gender"]=le.fit_transform(data1["gender"])
data1["ssc_b"]=le.fit_transform(data1["ssc_b"])
data1["hsc_b"]=le.fit_transform(data1["hsc_b"])
data1["hsc_s"]=le.fit_transform(data1["hsc_s"])
data1["degree_t"]=le.fit_transform(data1["degree_t"])
data1["workex"]=le.fit_transform(data1["workex"])
data1["specialisation"]=le.fit_transform(data1["specialisation"] )     
data1["status"]=le.fit_transform(data1["status"])       
data1 

x=data1.iloc[:,:-1]
x

y=data1["status"]
y

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LogisticRegression
lr=LogisticRegression(solver="liblinear")
lr.fit(x_train,y_train)
y_pred=lr.predict(x_test)
y_pred

from sklearn.metrics import accuracy_score
accuracy=accuracy_score(y_test,y_pred)
accuracy

from sklearn.metrics import confusion_matrix
confusion=confusion_matrix(y_test,y_pred)
confusion

from sklearn.metrics import classification_report
classification_report1 = classification_report(y_test,y_pred)
print(classification_report1)

lr.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])

```

## Output:
# Placement data:
<img width="816" height="685" alt="313921198-19b35163-7efc-4c7d-aa80-3b8a68fe16ce" src="https://github.com/user-attachments/assets/4da87201-069b-4130-ac59-5ce3b3bf1456" />


# Salary data:


<img width="1250" height="405" alt="Screenshot 2025-09-16 111324" src="https://github.com/user-attachments/assets/d93d7f3b-464d-43bf-8421-5592deb88aa3" />

# Checking the null() function:

<img width="549" height="393" alt="Screenshot 2025-09-16 111231" src="https://github.com/user-attachments/assets/ac19f4b3-2883-41f6-9a98-33e2e19ddab2" />

# Data duplicate:
<img width="483" height="237" alt="Screenshot 2025-09-16 111224" src="https://github.com/user-attachments/assets/fa5012b2-5d2b-487d-a75b-46e70c4072f5" />


# Print data:

<img width="1249" height="522" alt="Screenshot 2025-09-16 111147" src="https://github.com/user-attachments/assets/d5f5d1c9-0be4-4b6f-b8e8-969c45f955ff" />

# Data-status:

<img width="365" height="588" alt="Screenshot 2025-09-16 111132" src="https://github.com/user-attachments/assets/4ce1bc30-5eb2-46fa-b2d6-cbfb5e6d90aa" />

# Y_prediction array:

<img width="968" height="354" alt="Screenshot 2025-09-16 111122" src="https://github.com/user-attachments/assets/d1dbc435-6faf-4163-82dc-04baa87ebc95" />

# Accuracy value:
<img width="320" height="68" alt="Screenshot 2025-09-16 111108" src="https://github.com/user-attachments/assets/477995ea-a71e-45a7-b2f3-c7124c583839" />


# Confusion array:

<img width="497" height="85" alt="Screenshot 2025-09-16 111103" src="https://github.com/user-attachments/assets/f7971f8e-a0cf-4368-9917-37511c2cb49a" />

# Classification Report:

<img width="792" height="325" alt="Screenshot 2025-09-16 111056" src="https://github.com/user-attachments/assets/22f36c9c-320f-4bf5-bc15-b751a1ebbb49" />

# Prediction of LR:
<img width="665" height="273" alt="Screenshot 2025-09-16 111047" src="https://github.com/user-attachments/assets/d872220d-6fbb-4e4d-b631-53b3c762893e" />


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
