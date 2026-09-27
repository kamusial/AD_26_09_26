# regresja

import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

df = pd.read_csv('f-c.csv', usecols=[1, 2])

model = Sequential()
model.add(Dense(1, activation='linear'))
model.add(Dense(4, activation='linear'))
model.add(Dense(4, activation='relu'))
model.add(Dense(1, activation='linear'))

model.compile(optimizer='rmsprop', loss='mse')

result = model.fit(df.F, df.C, epochs=2000)

C_pred = model.predict(df.F)

plt.scatter(df.F, df.C)
plt.plot(df.F, C_pred,c='r')
plt.show()
