import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#Importing the dataset
dataset = pd.read_csv('Data.csv')

X = dataset.iloc[:,:-1].values
dx = pd.DataFrame(X)
y = dataset.iloc[:,-1].values
dy = pd.DataFrame(y)

#Insert the missing values
from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN' , strategy='mean' , axis = 0)

X[:,1:3] = imputer.fit_transform(X[:,1:3])

#Categorising Data
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder_X = LabelEncoder()
labelencoder_y = LabelEncoder()

X[:,0] = labelencoder_X.fit_transform(X[:,0])
y = labelencoder_y.fit_transform(y)

onehotencoder_X = OneHotEncoder(categorical_features= [0])
X = onehotencoder_X.fit_transform(X).toarray()

#Splitting data into training and test set

from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test  = train_test_split(X,y,test_size = 0.2, random_state = 0)

#Normalization
from sklearn.preprocessing import StandardScaler
sc_X  =StandardScaler()
X_train = sc_X.fit_transform(X_train)
X_test = sc_X.transform(X_test)