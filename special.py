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

def tanh_crossentropy(y_true,y_pred,batch_repeats):
    y_true = K.repeat_elements(y_true,batch_repeats,axis=0)
    return K.sum(binary_crossentropy((y_true+1)/2,(y_pred+1)/2))

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
            
            geom_val = self.geom_drop_layer.get_geom_val()+1
            print('updating geom dropout indices, now at :',geom_val)
            self.geom_drop_layer.set_geom_val(geom_val)

    def get_monitor_value(self, logs):
        monitor_value = logs.get(self.monitor)
        if monitor_value is None:
            warnings.warn(
                'Early stopping conditioned on metric `%s` '
                'which is not available. Available metrics are: %s' %
                (self.monitor, ','.join(list(logs.keys()))), RuntimeWarning
            )
        return monitor_value


def build_latent_block(input_shape,geom_rate,num_repeats=1):
    input_layer = keras.layers.Input(shape=input_shape,name='latent_input')
    sampled = BernoulliSampling(num_repeats)(input_layer)
    tanh = Lambda(lambda x : (x-0.5)*2)(sampled)
    drop = GeometricDropout(geom_rate,name='geom_dropout')(tanh)
    return keras.models.Model([input_layer],[drop],name='latent_block')
    
class BernoulliSampling(Layer):

    def __init__(self,num_repeats=1,**kwargs):
        super(BernoulliSampling, self).__init__(**kwargs)
        self.supports_masking = True
        self.num_repeats = num_repeats


    def call(self, inputs, training=None):
        _,latent_size = inputs.shape.as_list()
        def sampled():
            dist = tfp.distributions.Bernoulli(probs=inputs,dtype=tf.float32)
            sample =  dist.sample(sample_shape=(self.num_repeats))
            s_shape = tf.shape(inputs)
            out = tf.reshape(sample,(s_shape[0]*self.num_repeats,latent_size))
            return out
        
        def quantized():
            differentiable_round = tf.maximum(inputs-0.499,0)
            differentiable_round = differentiable_round * 10000
            differentiable_round = tf.minimum(differentiable_round, 1)
#            return differentiable_round
            return K.repeat_elements(differentiable_round,self.num_repeats,axis=0)
#            return K.stop_gradient(K.round(inputs)) #why doesn't this work
        
        return K.in_train_phase(sampled, quantized, training=training)

    def get_config(self):
        config = {'num_repeats': self.num_repeats}
        base_config = super(BernoulliSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
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
        self.set_geom_indices()
        
    def get_geom_val(self):
        return self.geom_val
    
    def set_geom_indices(self):
        _indices = np.expand_dims(np.arange(0,self.latent_size)-self.geom_val,axis=0)
        self.set_weights([_indices])
        
    def build(self,input_shape):
        self.latent_size = input_shape.as_list()[1]
        self.indices = self.add_weight(name="geom_indices",
                                     shape=(1,self.latent_size,),
                                     dtype=tf.float32,
                                     trainable=False)
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

class RepeatBatch(Layer):

    def __init__(self, num_repeats,**kwargs):
        super(RepeatBatch,self).__init__(**kwargs)
        self.supports_masking = True
        self.num_repeats = num_repeats


    def call(self, inputs, training=None):

        return K.repeat_elements(inputs, self.num_repeats, axis=0)

    def get_config(self):
        config = {'num_repeats': self.num_repeats,}
        base_config = super(RepeatBatch, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
def build_repeat_block(input_shape,num_repeats):
    input_layer = keras.layers.Input(shape=input_shape,name='repeat_input')
    
    out = RepeatBatch(num_repeats)(input_layer)

    return keras.models.Model([input_layer],[out],name='repeat_block')


if __name__ == '__main__':
    sess = K.get_session()
    num_samples = 50
    indices = np.expand_dims(np.arange(0,100),axis=0)
    indices = np.repeat(indices,num_samples,axis=0)
    indices = tf.convert_to_tensor(indices,dtype=tf.float32)
    geom = tfp.distributions.Geometric(probs=[0.03]).sample(sample_shape=(num_samples))
    drop = tf.cast(indices <= geom,tf.uint8)
    out = sess.run(fetches=[drop,geom])
    