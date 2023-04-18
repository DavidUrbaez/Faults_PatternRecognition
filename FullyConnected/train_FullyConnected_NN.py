import h5py
from pathlib import Path
import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split

data_path_3pg = Path(r"H:\datasets\Sim3_3fg_Randt_datos_fallas.h5")
data_path_1pg = Path(r"H:\datasets\Sim3_1fg_Randt_datos_fallas.h5")

print("read data")

with h5py.File(data_path_1pg, 'r') as hf:
    # hf.keys()
    t = np.array(hf.get('time'))
    data_1pg = np.array(hf.get('data'))  # --> y=0

with h5py.File(data_path_3pg, 'r') as hf:
    data_3pg = np.array(hf.get('data'))  # --> y=1

y_1pg = np.zeros((data_1pg.shape[0], 1))
y_3pg = np.ones((data_3pg.shape[0], 1))

X = np.concatenate([data_1pg[:, :, 3:], data_3pg[:, :, 3:]], axis=0)
y = np.concatenate([y_1pg, y_3pg], axis=0)
X = X.reshape(X.shape[0], X.shape[1] * X.shape[2]) / 200

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=10)

# Neural network
model = Sequential()
model.add(Dense(100, input_dim=3003, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(100, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test), )
model.save('models/FullyConnected.h5')


