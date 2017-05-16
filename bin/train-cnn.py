import h5py
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras.layers import Conv3D, MaxPool3D, Flatten, Dense, Dropout, Input, BatchNormalization
from keras.models import Model

# load the data
with h5py.File('full_dataset_vectors.h5', 'r') as hf:
    x_train = hf["X_train"][:]
    y_train = hf["y_train"][:]
    x_test = hf["X_test"][:]
    y_test = hf["y_test"][:]

# translate the data to color
def array_to_color(array, cmap="Oranges"):
    s_m = plt.cm.ScalarMappable(cmap=cmap)
    return s_m.to_rgba(array)[:,:-1]

# translate the train data
xtrain = np.ndarray((x_train.shape[0], 4096, 3))
for i in range(x_train.shape[0]):
    xtrain[i] = array_to_color(x_train[i])


# translate the test data
xtest = np.ndarray((x_test.shape[0], 4096, 3))
for i in range(x_test.shape[0]):
    xtest[i] = array_to_color(x_test[i])



# keras inputdata should be 5D shape, so we shoule reshape the data
xtrain = xtrain.reshape(10000, 16, 16, 16, 3)
xtest = xtest.reshape(2000, 16, 16, 16, 3)

# we have 10 labels, we use keras translate labels to one-hot
y_train = keras.utils.to_categorical(y_train, 10)
y_test = keras.utils.to_categorical(y_test, 10)

# get the cnn model
def get_model():
    ins = Input((16, 16, 16, 3))
    con1 = Conv3D(filters=8, kernel_size=(3, 3, 3), activation='relu')(ins)
    con2 = Conv3D(filters=16, kernel_size=(3, 3, 3), activation='relu')(con1)
    maxp3 = MaxPool3D(pool_size=(2, 2, 2))(con2)
    con4 = Conv3D(filters=32, kernel_size=(3, 3, 3), activation='relu')(maxp3)
    con5 = Conv3D(filters=64, kernel_size=(3, 3, 3), activation='relu')(con4)
    maxp6 = MaxPool3D(pool_size=(2, 2, 2))(con2)
    batch = BatchNormalization()(maxp6)
    flat = Flatten()(batch)
    dens1 = Dense(units=4096, activation='relu')(flat)
    drop1 = Dropout(0.7)(dens1)
    dens2 = Dense(units=1024, activation='relu')(drop1)
    drop2 = Dropout(0.7)(dens2)
    outs = Dense(units=10, activation='softmax')(drop2)
    model = Model(inputs=ins, outputs=outs)
    model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(lr=0.1), metrics=['accuracy'])
    
    return model

model = get_model()

# train our model
model.fit(x=xtrain[:8000], y=y_train[:8000], batch_size=10, epochs=50, validation_split=0.2)