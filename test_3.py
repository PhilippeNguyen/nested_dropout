import tensorflow as tf
import tensorflow_probability as tfp
from tensorflow_probability import distributions as tfd
from tensorflow_probability import bijectors as tfb
from tensorflow import keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.datasets.mnist import load_data
import numpy as np
import matplotlib.pyplot as plt
import argparse
from distutils.util import strtobool
from special import BernoulliSampling

sess= K.get_session()


#generate data
((img_train,class_train),(img_test,class_test)) = load_data()
x_train = np.expand_dims((img_train-127.5)/127.5,3)
x_test = np.expand_dims((img_test-127.5)/127.5,3)
y_train = K.one_hot(class_train,10).eval(session=sess)
y_test = K.one_hot(class_test,10).eval(session=sess)

out_size = 10
data_shape = x_train.shape[1:]
    
##Build model
#hmm, with very small model it works,
#but with two hidden layers it doesn't
input_layer = keras.layers.Input(shape=data_shape)
x = input_layer
x = keras.layers.Conv2D(64,kernel_size=3,activation='sigmoid')(x)
x = keras.layers.AveragePooling2D(2)(x)
x = keras.layers.Conv2D(64,kernel_size=3,activation='sigmoid')(x)

x = keras.layers.AveragePooling2D(2)(x)
params = keras.layers.Flatten()(x)
bern = BernoulliSampling(1.9)(params)

x = keras.layers.Dense(out_size,activation='softmax')(bern)

model = keras.models.Model([input_layer],[x])
model_output = model.output
params_model = keras.models.Model([input_layer],[params])
bern_model = keras.models.Model([input_layer],[bern])
#
model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
              loss='categorical_crossentropy')   

#train model 
early_stop = keras.callbacks.EarlyStopping(patience=5)
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),
          epochs=500,batch_size=512,callbacks=[early_stop])

out = model.predict(x_test)
params_out = params_model.predict(x_test)
bern_out = bern_model.predict(x_test)