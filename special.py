import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer,Lambda,Multiply
import tensorflow.keras.backend as K
#import keras.backend as K
import numpy as np


from tensorflow.keras.losses import binary_crossentropy

def tanh_crossentropy(y_true,y_pred):
    return keras.backend.sum(binary_crossentropy((y_true+1)/2,(y_pred+1)/2))


def build_latent_block(input_shape,geo_rate):
    input_layer = keras.layers.Input(shape=input_shape,name='latent_input')
    sampled = BernoulliSampling()(input_layer)
    tanh = Lambda(lambda x : (x-0.5)*2)(sampled)
    drop = GeometricDropout(geo_rate)(tanh)
    return keras.models.Model([input_layer],[drop],name='latent')
#    return keras.models.Model([input_layer],[tanh],name='latent')

class BernoulliSampling(Layer):

    def __init__(self, **kwargs):
        super(BernoulliSampling, self).__init__(**kwargs)
        self.supports_masking = True


    def call(self, inputs, training=None):

        def sampled():
            dist = tfp.distributions.Bernoulli(probs=inputs,dtype=tf.float32)
            return dist.sample()
        
        def quantized():
            return K.round(inputs)
        
        return K.in_train_phase(sampled, quantized, training=training)


    def get_config(self):
        return super(BernoulliSampling, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape

class GeometricDropout(Layer):

    def __init__(self, rate, **kwargs):
        super(GeometricDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate

    def call(self, inputs, training=None):
        if 0 < self.rate < 1:
            def noised():
                #build_indices
                latent_size = inputs.shape.as_list()[1]
                indices = np.expand_dims(np.arange(0,latent_size),axis=0)
                indices = tf.convert_to_tensor(indices,dtype=tf.float32)
                
                indices = tf.broadcast_to(indices,K.shape(inputs))
                geom = tfp.distributions.Geometric(probs=[self.rate])
                geom_sample = geom.sample(sample_shape=(K.shape(inputs)[0]))
                drop = tf.cast(indices <= geom_sample,tf.float32)
                return inputs * drop
            
            return K.in_train_phase(noised, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(GeometricDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
    #TODO have indices be a variable that can be updated
    def set_indices(self):
        pass

if __name__ == '__main__':
    sess = K.get_session()
    num_samples = 50
    indices = np.expand_dims(np.arange(0,100),axis=0)
    indices = np.repeat(indices,num_samples,axis=0)
    indices = tf.convert_to_tensor(indices,dtype=tf.float32)
    geom = tfp.distributions.Geometric(probs=[0.03]).sample(sample_shape=(num_samples))
    drop = tf.cast(indices <= geom,tf.uint8)
    out = sess.run(fetches=[drop,geom])
    