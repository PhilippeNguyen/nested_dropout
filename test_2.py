

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
def get_loss(dist,squeeze=True):
    def loss (y_true,y_pred):
        if squeeze:
            y_true = tf.squeeze(y_true,axis=1) #needed for keras/tfp bug fix?
        neg_log_likelihood = -tf.reduce_mean(dist.log_prob(y_true))
        return neg_log_likelihood
    return loss

sess= K.get_session()


#generate data
((img_train,class_train),(img_test,class_test)) = load_data()

x_train = K.one_hot(class_train,10).eval(session=sess)
x_test = K.one_hot(class_test,10).eval(session=sess)
y_train = np.zeros(x_train.shape[0])
y_train[x_train[:,0]==1] =1
y_test = np.zeros(x_test.shape[0])
y_test[x_test[:,0]==1] =1
out_size = 1
data_shape = x_train.shape[1:]
    
##Build model
input_layer = keras.layers.Input(shape=data_shape)
params = keras.layers.Dense(out_size,activation='sigmoid')(input_layer)
x = BernoulliSampling(0.1)(params)
model = keras.models.Model([input_layer],[x])
model_output = model.output
params_model = keras.models.Model([input_layer],[params])
#
model.compile(optimizer=keras.optimizers.Adam(lr=0.01),
              loss='binary_crossentropy')   

#train model 
early_stop = keras.callbacks.EarlyStopping(patience=5)
model.fit(x=x_train,y=y_train,validation_data=(x_test,y_test),
          epochs=500,batch_size=512,callbacks=[early_stop])

out = params_model.predict(x_test)
