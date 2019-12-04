import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

dataset=pd.read_csv('subdataset(1000data1).csv')
#print 1st 20 details
print(dataset.head(20))
print(dataset.info())
print(dataset.describe())
#no of rows
print(len(dataset))
#no of colmns
print(len(dataset.columns))

print(dataset.shape)

#Analysis of data how they looks like on graphical representation
#graph himidity vs temparture 
plt.scatter(np.log10(dataset[' _hum']),dataset[' _tempm'])
plt.title('humidity vs temprature')
plt.xlabel("------------humidity-------------")
plt.ylabel("------------temprature-----------")
plt.show()
#histogram of data how they looks like on graphical represantation
plt.hist(dataset[' _tempm'],facecolor='red',edgecolor='blue',bins=10,range=(5,35))
plt.title("temprature histogram")
plt.show()
#graph pressure vs temparture 
plt.scatter(np.log10(dataset[' _pressurem']),dataset[' _tempm'])
plt.title('pressure vs temprature')
plt.xlabel("------------pressure-------------")
plt.ylabel("------------temprature-----------")
plt.show()
#histogram of data how they looks like on graphical represantation
plt.hist(dataset[' _pressurem'],facecolor='green',edgecolor='red',bins=10,range=(1000,1025))
plt.title("pressure histogram")
plt.show()

#graph dewpoint vs temparture 
plt.scatter(np.log10(dataset[' _dewptm']),dataset[' _tempm'])
plt.title(' dewpoint vs temprature')
plt.xlabel("----------- dewpoint-------------")
plt.ylabel("------------temprature-----------")
plt.show()
#histogram of data how they looks like on graphical represantation
plt.hist(dataset[' _dewptm'],facecolor='pink',edgecolor='red',bins=10,range=(0,25))
plt.title(" dewpoint histogram")
plt.show()

#data wrangling
print(dataset.isnull())
#table of content in terms true and false
print(dataset.isnull().sum())


#droping all unuseful column

dataset.drop([" _heatindexm"],axis=1,inplace=True)
dataset.drop([" _precipm"],axis=1,inplace=True)
#output Delete all null values
print(dataset.isnull().sum())
dataset.dropna(inplace=True)
#check is there any null value
print(dataset.head(20))
print(dataset.isnull().sum())

dataset.drop(["datetime_utc"],axis=1,inplace=True)
#delete all values from the pressure which has a value -9999
indexn=dataset[dataset[' _pressurem']==-9999].index
dataset.drop(indexn,inplace=True)
#taking all the features into x variable  and y for prediction
Y=dataset.iloc[:,len(dataset.columns)-1]
X=dataset.iloc[:,0:len(dataset.columns)-1]

#set the dummies value as a level for the weather clacification
weather_condition=pd.get_dummies(X[' _conds'])
#delete last dummies value which is null
weather_condition.drop(["Unknown"],axis=1,inplace=True)
print(weather_condition.head(10))
#concat the dummies value with the input feature X
X=pd.concat([X,weather_condition],axis=1)

print(X.head(10))
X.drop([" _conds"],axis=1,inplace=True)
print(X.shape)
#now final data set has been created
print(X.head(10))

#####################################################################

# train and testing

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

#splitting Dataset into train set and test set
X_train,X_test,y_train,y_test=train_test_split(X,Y,test_size=0.2,random_state=0)
model=LinearRegression()
model.fit(X_train,y_train)

y_prediction=model.predict(X_test)

score=r2_score(y_test,y_prediction)

print("Temprature prediction Accuracy= ",score*100)

#histogram of data how they looks like on graphical represantation
plt.hist(y_prediction,facecolor='red',edgecolor='blue',bins=10,range=(5,35))
plt.title("predicted temprature histogram")
plt.show()


plt.hist(y_test,facecolor='red',edgecolor='blue',bins=10,range=(5,35))
plt.title("dataset temprature histogram")
plt.show() 
# original temprature vs predicted temprature graph
plt.plot(y_test)
plt.plot(y_prediction)
plt.title('original temprature vs temprature')
plt.xlabel("------------x-axis-------------")
plt.ylabel("------------y-axis-----------")
plt.legend()
plt.show()


from scipy.special import softmax 
#splitting Dataset into train set and test set
X1_train,X1_test,y1_train,y1_test=train_test_split(X,weather_condition,test_size=0.2,random_state=0)
model1=LinearRegression()
model1.fit(X1_train,y1_train)

y1_prediction=model1.predict(X1_test)
predicted_probabilities=softmax(y1_prediction)
print(y1_prediction)
#print predicted probabilities of each clasaes
print(predicted_probabilities)      
score1=r2_score(y1_test,y1_prediction)

print("weather  clacification prediction Accuracy= ",score1*100)

#histogram of data how they looks like on graphical represantation
plt.hist(y1_prediction,facecolor='red',edgecolor='blue',bins=10,range=(0,1))
plt.title("predicted weather  clacification histogram")
plt.show()


plt.hist(y1_test,facecolor='red',edgecolor='blue',bins=10,range=(0,1))
plt.title("datase tweather  clacification histogram")
plt.show() 
