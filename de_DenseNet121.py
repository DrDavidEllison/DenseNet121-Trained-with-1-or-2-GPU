import numpy as np
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import multi_gpu_model

import time
import os

h5file = os.path.abspath(os.curdir)+'/densenet121_weights_tf_dim_ordering_tf_kernels.h5'
#print(h5file)
model = keras.applications.densenet.DenseNet121(include_top=True, weights=h5file, input_tensor=None, input_shape=None, pooling=None, classes=1000)
#model = keras.applications.densenet.DenseNet121(include_top=True, weights='/linkhome/idris/sos/ssos189/workdir/AIBench/bench-tf-keras/benchmarks-keras/densenet121_weights_tf_dim_ordering_tf_kernels.h5', input_tensor=None, input_shape=None, pooling=None, classes=1000)


channels = 3
dx = 224
dy = 224
minibatch = 32

# Generate dummy data
x_train = np.random.random((640, dx, dy, channels))
y_train = keras.utils.to_categorical(np.random.randint(1000, size=(640, 1)), num_classes=1000)
x_test = np.random.random((640, dx, dy, channels))
y_test = keras.utils.to_categorical(np.random.randint(1000, size=(640, 1)), num_classes=1000)

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

model.fit(x_train, y_train, batch_size=minibatch, epochs=1)

t1 = time.time ()

model.fit(x_train, y_train, batch_size=minibatch, epochs=2)

t2 = time.time ()

parallel_model = multi_gpu_model(model, gpus=2)
parallel_model.compile(loss='categorical_crossentropy', optimizer=sgd)

parallel_model.fit(x_train, y_train, batch_size=minibatch, epochs=1)

t7 = time.time ()

parallel_model.fit(x_train, y_train, batch_size=minibatch, epochs=2)

t8 = time.time ()

print ()
print ("Single GPU train DenseNet121 with Keras in ", t2 - t1, "seconds")
print ()
print ("Two GPU train DenseNet121 with Keras in ", t8 - t7, "seconds")
print ()
