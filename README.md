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
Developed by: Reklies J
RegisterNumber: 212223110041
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

<img width="816" height="685" alt="313921198-19b35163-7efc-4c7d-aa80-3b8a68fe16ce" src="https://github.com/user-attachments/assets/18e2f897-be6b-4c4d-962e-b6022a3b87c0" />

# Salary data:

<img width="1250" height="405" alt="Screenshot 2025-09-16 111324" src="https://github.com/user-attachments/assets/7452d921-371d-4c91-a5cd-550946603b03" />


# Checking the null() function:

<img width="549" height="393" alt="Screenshot 2025-09-16 111231" src="https://github.com/user-attachments/assets/9627f6b4-0be5-4368-b4a9-c91957f341b5" />

# Data duplicate:

<img width="483" height="237" alt="Screenshot 2025-09-16 111224" src="https://github.com/user-attachments/assets/5af41882-3827-4aa1-908e-d59c8c811d05" />

# Print data:

<img width="1249" height="522" alt="Screenshot 2025-09-16 111147" src="https://github.com/user-attachments/assets/33268671-134a-4e91-abcc-fb388567fe86" />

# Data-status:

<img width="365" height="588" alt="Screenshot 2025-09-16 111132" src="https://github.com/user-attachments/assets/9da2f1ce-a373-4efa-a1d4-eb49ae885976" />

# Y_prediction array:

<img width="968" height="354" alt="Screenshot 2025-09-16 111122" src="https://github.com/user-attachments/assets/09f471b5-331d-4dc8-bf03-1ef4e255d1ae" />

# Accuracy value:
<img width="320" height="68" alt="Screenshot 2025-09-16 111108" src="https://github.com/user-attachments/assets/ce8cb4b7-d15a-48c0-80f3-baa4cf1803ea" />


# Confusion array:

<img width="497" height="85" alt="Screenshot 2025-09-16 111103" src="https://github.com/user-attachments/assets/e49551f8-bc9f-4442-a978-69107413c9aa" />

# Classification Report:

<img width="792" height="325" alt="Screenshot 2025-09-16 111056" src="https://github.com/user-attachments/assets/a481a68e-9bc6-43f8-8816-938a635a55d3" />

# Prediction of LR:

<img width="665" height="273" alt="Screenshot 2025-09-16 111047" src="https://github.com/user-attachments/assets/d9d21204-493e-4468-a3e0-713ca89a3ca3" />

## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
