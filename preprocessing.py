import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split



data = pd.read_csv('D:\AI Course\tasks\task02\Churn_Modelling.csv')
print(data)


features = data.iloc[:,0:-1]
features = pd.get_dummies(features)
print(features)

labels = data.iloc[:-1,-1:]
labels = pd.get_dummies(labels)
print(labels)



imputer = SimpleImputer()
features = imputer.fit_transform(features)
print(features)


scaler = MinMaxScaler()
features = imputer.fit_transform(features)
print(features)


encode = LabelEncoder()
features = imputer.fit_transform(labels)
print(labels)


x_train,y_train,x_test,y_test = train_test_split(features,labels,test_size=0.2)
print(x_train)
print(y_train)
print(x_test)
print(y_test)
