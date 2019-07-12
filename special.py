import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer,Lambda,Multiply
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy

#import  keras
#import keras.backend as K
#from keras.layers import Layer,Lambda,Multiply
#from keras.callbacks import Callback,ModelCheckpoint
#from keras.losses import binary_crossentropy

#from keras.losses import binary_crossentropy

import numpy as np
import warnings
from tensorflow.python.platform import tf_logging as logging




          
#def tanh_crossentropy(y_true,y_pred,batch_repeats):
##    y_true = K.repeat_elements(y_true,batch_repeats,axis=0)
#    y_true = tf.tile(y_true,batch_repeats)
#    return K.sum(binary_crossentropy((y_true+1)/2,(y_pred+1)/2))

def tanh_crossentropy(y_true,y_pred):
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
    
#    
class FixedModelCheckpoint(ModelCheckpoint):
    def __init__(self, filepath, save_model,**kwargs):
        super(FixedModelCheckpoint, self).__init__(filepath,**kwargs)
        self.save_model = save_model
        
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch + 1, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % (self.monitor), RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch + 1, self.monitor, self.best,
                                     current, filepath))
                        self.best = current
                        if self.save_weights_only:
                            self.save_model.set_weights(self.model.get_weights())
                            self.save_model.save_weights(filepath, overwrite=True)
                        else:
                            self.save_model.set_weights(self.model.get_weights())
                            self.save_model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('\nEpoch %05d: %s did not improve from %0.5f' %
                                  (epoch + 1, self.monitor, self.best))
            else:
                if self.verbose > 0:
                    print('\nEpoch %05d: saving model to %s' % (epoch + 1, filepath))
                if self.save_weights_only:
                    self.save_model.set_weights(self.model.get_weights())
                    self.save_model.save_weights(filepath, overwrite=True)
                else:
                    self.save_model.set_weights(self.model.get_weights())
                    self.save_model.save(filepath, overwrite=True)


def build_latent_block(input_shape,geom_rate,
                       temperature=0.1,use_grad_stop_mask=True,
                       sampling=True,dropout=True):
    input_layer = keras.layers.Input(shape=input_shape,name='latent_input')
    
    if sampling:
        x = BernoulliSampling(temperature=temperature)(input_layer)
    else:
        x = input_layer
        
    tanh = Lambda(lambda x : (x-0.5)*2)(x)
    
    if dropout:
        out = GeometricDropout(geom_rate,name='geom_dropout',
                                use_grad_stop_mask=use_grad_stop_mask)(tanh)
    else:
        out = tanh
        
    return keras.models.Model([input_layer],[out],name='latent_block')
    
class BernoulliSampling(Layer):

    def __init__(self,temperature=0.1,**kwargs):
        super(BernoulliSampling, self).__init__(**kwargs)
        self.supports_masking = True
        self.temperature = temperature


    def call(self, inputs, training=None):
        _,latent_size = inputs.shape.as_list()
        def sampled():
            dist = tfp.distributions.RelaxedBernoulli(probs=inputs,
                                                      temperature=self.temperature)
            sample =  dist.sample(sample_shape=())
            return sample
            

        
        def quantized():
            differentiable_round = tf.maximum(inputs-0.499,0)
            differentiable_round = differentiable_round * 10000
            differentiable_round = tf.minimum(differentiable_round, 1)
            return differentiable_round

        
        return K.in_train_phase(sampled, quantized, training=training)

    def get_config(self):
        config = {'temperature':self.temperature}
        base_config = super(BernoulliSampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
    def compute_output_shape(self, input_shape):
        return input_shape

class GeometricDropout(Layer):

    def __init__(self, rate, geom_val=0,use_grad_stop_mask=True, **kwargs):
        super(GeometricDropout, self).__init__(**kwargs)
        self.supports_masking = True
        self.rate = rate
        self.geom_val = geom_val
        self.latent_size = None
        self.indices = None
        self.stop_gradient_mask = None
        self.gradient_mask = None
        self.valid_mask = None
        self.use_grad_stop_mask = use_grad_stop_mask
        
    def set_geom_val(self,geom_val):
        self.geom_val = geom_val
        self.set_geom_indices()
        
    def get_geom_val(self):
        return self.geom_val
    
    def set_geom_indices(self):
        _indices = np.expand_dims(np.arange(0,self.latent_size)-self.geom_val,axis=0)
        stop_gradient_mask = np.zeros(((1,self.latent_size,)))
        stop_gradient_mask[...,:self.geom_val] = 1
        gradient_mask = np.ones(((1,self.latent_size,)))
        gradient_mask[...,:self.geom_val] = 0
        valid_mask = np.zeros(((1,self.latent_size,)))
        valid_mask[...,:self.geom_val+1] = 1
        self.set_weights([_indices,stop_gradient_mask,gradient_mask,valid_mask])
        
    def build(self,input_shape):
        self.latent_size = input_shape.as_list()[1]
        self.indices = self.add_weight(name="geom_indices",
                                     shape=(1,self.latent_size,),
                                     dtype=K.floatx(),
                                     trainable=False)
        self.stop_gradient_mask = self.add_weight(
                                name="stop_gradient_mask",
                                 shape=(1,self.latent_size,),
                                 dtype=K.floatx(),
                                 trainable=False)
        self.gradient_mask = self.add_weight(
                            name="gradient_mask",
                             shape=(1,self.latent_size,),
                             dtype=K.floatx(),
                             trainable=False)
        self.valid_mask = self.add_weight(
                        name="valid_mask",
                         shape=(1,self.latent_size,),
                         dtype=K.floatx(),
                         trainable=False)
        self.set_geom_indices()

        
    def call(self, inputs, training=None):
        
        if 0 < self.rate < 1:
            def noised():
                
                indices = tf.broadcast_to(self.indices,K.shape(inputs))
                geom = tfp.distributions.Geometric(probs=[self.rate])
                geom_sample = geom.sample(sample_shape=(K.shape(inputs)[0]))
                drop = tf.cast(indices <= geom_sample,tf.float32)
                
                if self.use_grad_stop_mask:
                    stop_gradient_mask = tf.broadcast_to(self.stop_gradient_mask,K.shape(inputs))
                    gradient_mask= tf.broadcast_to(self.gradient_mask,K.shape(inputs))
                    
                    masked_gradient_input = tf.stop_gradient(inputs*stop_gradient_mask) + inputs*gradient_mask
                    return masked_gradient_input * drop
                else:
                    return inputs * drop
            
            def valid_masked_only():
                valid_mask = tf.broadcast_to(self.valid_mask,K.shape(inputs))
                return inputs*valid_mask
            
            return K.in_train_phase(noised, valid_masked_only, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,'geom_val':self.geom_val,
                  'use_grad_stop_mask':self.use_grad_stop_mask}
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
#        return inputs
#        print('hello')
#        print(inputs.shape)
#        multiples = [1]*len(inputs.shape)
#        multiples[0] = self.num_repeats
#        print(multiples)
#        return tf.tile(inputs,multiples)
        

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


custom_objs = globals()

if __name__ == '__main__':
    pass
#    sess = K.get_session()
#    num_samples = 50
#    indices = np.expand_dims(np.arange(0,100),axis=0)
#    indices = np.repeat(indices,num_samples,axis=0)
#    indices = tf.convert_to_tensor(indices,dtype=tf.float32)
#    geom = tfp.distributions.Geometric(probs=[0.03]).sample(sample_shape=(num_samples))
#    drop = tf.cast(indices <= geom,tf.uint8)
#    out = sess.run(fetches=[drop,geom])
#    