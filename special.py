import tensorflow as tf
import tensorflow_probability as tfp
import tensorflow.keras as keras
from tensorflow.keras.layers import Layer,Lambda,Multiply
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K
#import keras.backend as K
import numpy as np
import warnings

from tensorflow.keras.losses import binary_crossentropy

def tanh_crossentropy(y_true,y_pred):
    return keras.backend.sum(binary_crossentropy((y_true+1)/2,(y_pred+1)/2))

class UpdateGeomRate(Callback):

    def __init__(self,
                 geom_drop_layer,
                 monitor='val_loss',
                 verbose=0,
                 mode='auto',
                 ):
        super(UpdateGeomRate, self).__init__()
        self.geom_drop_layer = geom_drop_layer
        self.monitor = monitor
        self.verbose = verbose

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('EarlyStopping mode %s is unknown, '
                          'fallback to auto mode.' % mode,
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
        elif mode == 'max':
            self.monitor_op = np.greater
        else:
            if 'acc' in self.monitor:
                self.monitor_op = np.greater
            else:
                self.monitor_op = np.less



    def on_train_begin(self, logs=None):
        self.best = np.Inf if self.monitor_op == np.less else -np.Inf

    def on_epoch_end(self, epoch, logs=None):
        current = self.get_monitor_value(logs)
        if current is None:
            return

        if self.monitor_op(current, self.best):
            self.best = current
        else:
            geom_val = self.geom_drop_layer.get_geom_val()
            self.geom_drop_layer.set_geom_val(geom_val+1)

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


def build_latent_block(input_shape,geom_rate):
    input_layer = keras.layers.Input(shape=input_shape,name='latent_input')
    sampled = BernoulliSampling()(input_layer)
    tanh = Lambda(lambda x : (x-0.5)*2)(sampled)
    drop = GeometricDropout(geom_rate,name='geom_dropout')(tanh)
    return keras.models.Model([input_layer],[drop],name='latent')
    
#    tanh = Lambda(lambda x : (x-0.5)*2)(input_layer)
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

    def __init__(self, rate, geom_val=0, **kwargs):
        super(GeometricDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.geom_val = geom_val
        self.latent_size = None
        self.indices = None
        
    def set_geom_val(self,geom_val):
        self.geom_val = geom_val
        self.set_geom_indices
        
    def get_geom_val(self):
        return self.geom_val
    
    def set_geom_indices(self):
        _indices = np.expand_dims(np.arange(0,self.latent_size)-self.geom_val,axis=0)
        self.set_weights([_indices])
        
    def build(self,input_shape):
        self.latent_size = input_shape.as_list()[1]
        self.indices = self.add_weight(name="geom_indices",
                                     shape=(1,self.latent_size,),
                                     dtype=tf.float32)
        _indices = np.expand_dims(np.arange(0,self.latent_size)-self.geom_val,axis=0)
        self.set_weights([_indices])
    def call(self, inputs, training=None):
        
        if 0 < self.rate < 1:
            def noised():
                
                indices = tf.broadcast_to(self.indices,K.shape(inputs))
                geom = tfp.distributions.Geometric(probs=[self.rate])
                geom_sample = geom.sample(sample_shape=(K.shape(inputs)[0]))
                drop = tf.cast(indices <= geom_sample,tf.float32)
                return inputs * drop
            
            return K.in_train_phase(noised, inputs, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,'geom_val':self.geom_val}
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
    