# Recurrent Neural Network



# Part 1 - Data Preprocessing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#importing the tranning set
dataset_train = pd.read_csv("C:/Users/errk5/OneDrive/Desktop/RNN/Part 3 - Recurrent Neural Networks/Google_Stock_Price_Train.csv")
training_set = dataset_train.iloc[:,1:2].values

#Feature Scalimg
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range = (0,1))
training_set_scaled = sc.fit_transform(training_set)

#Creating a data structure with 60times steps and 1 output

X_train =[]
Y_train = []
for i in range(60,1258):
    X_train.append(training_set_scaled[i-60: i, 0])
    Y_train.append(training_set_scaled[i,0])
X_train, Y_train = np.array(X_train), np.array(Y_train)

#Reshaping
X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1],1))

#Part 2
#Inatlising the RNN
from  keras.models import Sequential
from  keras.layers import Dense
from  keras.layers import LSTM 
from  keras.layers import Dropout

#Initalising The RNN

regressor = Sequential()

#Adding the first LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True, input_shape =(X_train.shape[1],1)))
regressor.add(Dropout(0.2))

#Adding the second LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the third LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the fourth LSTM layer and some Dropout regularisation
regressor.add(LSTM(units=50, return_sequences=True))
regressor.add(Dropout(0.2))

#Adding the output layer
regressor.add(Dense(units=1))
#Compling the RNN
regressor.compile(optimizer='adam', loss = 'mean_squared_error')

#Fitting the RNN

regressor.fit(X_train,Y_train, epochs=100, batch_size=32)

#Part 3 MAking the prediction and visualising the result
#Getting the real stock price in 2017.
dataset_test = pd.read_csv('C:/Users/errk5/OneDrive/Desktop/deep learning/Part 3 - Recurrent Neural Networks/Google_Stock_Price_Train.csv')
real_stock_price = dataset_train.iloc[:,1:2].values

#Getting the predicted stock price of 2017
dataset_total = pd.concat((dataset_train['Open'], dataset_test['Open']), axis=0)
inputs = dataset_total[len(dataset_total) - len(dataset_test) - 60:].values
inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)
X_test = []
for i in range(60,80):
    X_test.append(inputs[i-60: i,0])
X_test = np.array(X_test)
X_test = np.array(X_test,(X_test.shape[0], X_test.shape[1],1))
predicted_stock_price = regressor.predicted_stock_price(X_test)
predicted_stock_price = sc.inverse_transform(predicted_stock_price)

#Visualising the results 
plt.plot(real_stock_price, color= 'red', label = 'Real Google stock price')
plt.plot(predicted_stock_price, color= 'blue', label = 'Predicted Google stock price')
plt.title('Google stock price prediction')
plt.xlabel('Time')
plt.ylabel('Google stock price')
plt.legend
plt.show()
