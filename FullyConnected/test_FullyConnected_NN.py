import h5py
from pathlib import Path
import numpy as np
from numpy import loadtxt
from keras.models import load_model

cwd = Path.cwd()
data_path_3pg = Path(r"H:\datasets\Sim3_3fg_Randt_datos_fallas.h5")
data_path_1pg = Path(r"H:\datasets\Sim3_1fg_Randt_datos_fallas.h5")
# data_path_3pg = Path(r"H:\datasets\3fg_datos_fallas.h5")
# data_path_1pg = Path(r"H:\datasets\1fg_datos_fallas.h5")
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

# load model
model = load_model('models/FullyConnected.h5')
# summarize model.
model.summary()
# load dataset

score = model.evaluate(X, y, verbose=0)
print("%s: %.2f%%" % (model.metrics_names[1], score[1] * 100))
