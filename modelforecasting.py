#import necessary packages
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pickle
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

#dataset on github
df=pd.read_csv("https://raw.githubusercontent.com/kruphacm/mini-project/main/forecasting%20dataset.csv")
#data preprocessing for forecasting 
df['BLOOD PRESSURE']=df['BLOOD PRESSURE'].str.split("/")
df.insert(0,"SYSTOLIC",df['BLOOD PRESSURE'].str[0])
df.insert(1,"DIASTOLIC",df['BLOOD PRESSURE'].str[1])
df['TEMPERATURE']=df['TEMPERATURE']
df['HEART BEAT']=df['HEART BEAT']
df['OXYGEN LEVEL']=df['OXYGEN LEVEL']
df['SYSTOLIC']=df['SYSTOLIC']
df['DIASTOLIC']=df['DIASTOLIC']
#HEARTBEAT CLASSIFICATION
#initialize X and Y
df['NORMAL(N) OR ABNORMAL(AN)'] = df['NORMAL(N) OR ABNORMAL(AN)'].map({'N':0.0,'AN':1.0,'N\n':0,'AN\n':1})
X=df[['HEART BEAT','HEART BEAT']]
Y=df['NORMAL(N) OR ABNORMAL(AN)']
#split data
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.20)
#create KNN model
heartbeat_model = KNeighborsClassifier(n_neighbors=4)
#fit model
heartbeat_model.fit(X_train,Y_train)
#generate pkl file
pickle.dump(heartbeat_model, open('modelheartbeat.pkl','wb'))

#OXYGEN  CLASSIFICATION
#initialize X and Y
X1=df[['OXYGEN LEVEL','OXYGEN LEVEL']]
Y1=df['NORMAL(N) OR ABNORMAL(AN)']
#split data
X1_train, X1_test, Y1_train, Y1_test = train_test_split(X1, Y1, test_size=0.20)
#create KNN model
oxygenlevel_model = KNeighborsClassifier(n_neighbors=4)
#fit the model
oxygenlevel_model.fit(X1_train,Y1_train)
#generate pkl file
pickle.dump(oxygenlevel_model, open('modeloxygen.pkl','wb'))

#SYSTOLIC AND DIASTOLIC CLASSIFICATION

#initialize X and Y
X2,Y2,X3,Y3=df[['SYSTOLIC','SYSTOLIC']],df['NORMAL(N) OR ABNORMAL(AN)'],df[['DIASTOLIC','DIASTOLIC']],df['NORMAL(N) OR ABNORMAL(AN)']
#Split data  
X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size=0.20)
#create model
systolic_model = KNeighborsClassifier(n_neighbors=4)
#fit data
systolic_model.fit(X2_train,Y2_train)
#generate pkl model
pickle.dump(systolic_model, open('modelsystolic.pkl','wb'))
#split data
X3_train, X3_test, Y3_train, Y3_test = train_test_split(X3, Y3, test_size=0.20)
#create model
diastolic_model = KNeighborsClassifier(n_neighbors=4)
#fit model
diastolic_model.fit(X3_train,Y3_train)
#generate pkl
pickle.dump(diastolic_model, open('modeldiastolic.pkl','wb'))

#TEMPERATURE CLASSIFICATION
#initialize x and y
X4=df[['TEMPERATURE','TEMPERATURE']]
Y4=df['NORMAL(N) OR ABNORMAL(AN)']
#split data
X4_train, X4_test, Y4_train, Y4_test = train_test_split(X4, Y4, test_size=0.20)
#create model
knn4 = KNeighborsClassifier(n_neighbors=4)
#fit model
knn4.fit(X4_train,Y4_train)
#generate pkl file
pickle.dump(knn4, open('modeltemperature.pkl','wb'))
#Forecasting models
#Temperature
training_set = df.iloc[:, 7:8].values
#MINIMISING VALUE TO RANGE 0 TO 1
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0, 1))
training_set_scaled = sc.fit_transform(training_set)
print(training_set_scaled)
X_train = []
y_train = []
for i in range(10,len(df['TEMPERATURE'])):# FORRECASTED BASED ON LAST 10 RECORDS
    X_train.append(training_set_scaled[i-10:i, 0])
    y_train.append(training_set_scaled[i, 0])
X_train, y_train = np.array(X_train), np.array(y_train) #(412,1)
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

#RNN MODEL
regressor = Sequential()
#INPUT LAYER
regressor.add(LSTM(units = 50, return_sequences = True, input_shape = (X_train.shape[1], 1)))
regressor.add(Dropout(0.2))
#HIDDEN LAYER 1
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
#HIDDEN LAYER 2
regressor.add(LSTM(units = 50, return_sequences = True))
regressor.add(Dropout(0.2))
# HIDDEN LAYER 3
regressor.add(LSTM(units = 50))
regressor.add(Dropout(0.2))
#OUTPUT LAYER
regressor.add(Dense(units = 1))

regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
regressor.fit(X_train, y_train, epochs = 80, batch_size = 32)
pickle.dump(regressor, open('forecast_temperature.sav','wb'))