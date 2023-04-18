import h5py
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
import tensorflow as tf

cwd = Path.cwd()
data_path_3pg = Path(r"H:\datasets\3fg_datos_fallas.h5")
data_path_1pg = Path(r"H:\datasets\1fg_datos_fallas.h5")
print("read data")
with h5py.File(cwd / data_path_1pg, 'r') as hf:
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

from sklearn.preprocessing import StandardScaler

# sc = StandardScaler()
# X = sc.fit_transform(X)

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)

# from sklearn.preprocessing import StandardScaler
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
# X_test = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)


# Neural network
model = Sequential()
model.add(Dense(100, input_dim=3003, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(100, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
              metrics=['accuracy'])

# X_train = np.expand_dims(X_train, axis=-1)
# X_test = np.expand_dims(X_test, axis=-1)

# train_data = tf.data.Dataset.from_tensor_slices((X_train, y_train))
# valid_data = tf.data.Dataset.from_tensor_slices((X_test, y_test))
# model.fit(train_data, epochs=10, validation_data=valid_data)
history = model.fit(X_train, y_train, epochs=10, batch_size=10, validation_data=(X_test, y_test), )
model.save('models/FullyConnected.h5')
x = 1
