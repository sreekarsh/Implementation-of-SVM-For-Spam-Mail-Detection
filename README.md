# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Import the required packages.
2.Import the dataset to operate on.
3.Split the dataset.
4.Predict the required output. 
```
## Program:
```
/*
Program to implement the SVM For Spam Mail Detection..
Developed by: RAGHUL V
RegisterNumber: 212223240132 
*/
import chardet
file='/content/spam.csv'
with open(file,'rb') as rawdata:
  result = chardet.detect(rawdata.read(100000))
result


import pandas as pd
data=pd.read_csv('/content/spam.csv',encoding='Windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```

## Output:
![image](https://github.com/sreekarsh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139841918/aa5a4795-3821-46ad-bc05-a11d8e2b3c7d)
![image](https://github.com/sreekarsh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139841918/fd703b23-f69d-43ca-8b7a-aa65ebd0172c)

![image](https://github.com/sreekarsh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139841918/1051695e-06e2-4609-adf0-a8a4029d6da7)
![image](https://github.com/sreekarsh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139841918/47fa2328-6c4f-4969-8af8-12c537a33663)

![image](https://github.com/sreekarsh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139841918/b86ef749-15d1-43f3-b9f8-951ede301f30)

![image](https://github.com/sreekarsh/Implementation-of-SVM-For-Spam-Mail-Detection/assets/139841918/9792fa2b-a7e9-4a99-a2ec-d187264e05ce)



## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
