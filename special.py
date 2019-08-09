import tensorflow as tf
import tensorflow_probability as tfp

import tensorflow.keras as keras
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Layer,Lambda,Multiply,Dropout,Dense,Activation
from tensorflow.keras.layers import Add
from tensorflow.keras.callbacks import Callback,ModelCheckpoint
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import initializers


import numpy as np
import warnings
from tensorflow.python.platform import tf_logging as logging
        
class TrainSequence(keras.utils.Sequence):

    def __init__(self, x_set, y_set, batch_size):
        self.x, self.y = x_set, y_set
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.x) / float(self.batch_size)))

    def __getitem__(self, idx):
        if self.y is None:
            batch_y = None
        else:
            batch_y = self.y[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch_x = self.x[idx * self.batch_size:(idx + 1) * self.batch_size]

        return batch_x,batch_y

def tanh_crossentropy(y_true,y_pred):
    bin_cross = binary_crossentropy((y_true+1)/2,(y_pred+1)/2)
    return K.mean(K.sum(bin_cross,axis=(1,2)))

def kl_from_uniform_bernoulli(latent_tensor,geo_index):
    #latent_tensor should have shape (batche_size,latent_size)
    base_dist = tfp.distributions.Bernoulli(logits=0.)
    latent_dist = tfp.distributions.Bernoulli(logits=latent_tensor[:,:geo_index])
    kl_dist = base_dist.kl_divergence(latent_dist)
    return kl_dist

class UpdateExtraParams(Callback):

    def __init__(self,
                 geom_drop_layer,
                 stop_grad_layer,
                 bern_layer,
                 monitor='val_loss',
                 verbose=0,
                 mode='auto',
                 update_count=3,
                 ):
        super(UpdateExtraParams, self).__init__()
        self.geom_drop_layer = geom_drop_layer
        self.stop_grad_layer = stop_grad_layer
        self.bern_layer = bern_layer
        self.monitor = monitor
        self.verbose = verbose
        self.count = 0
        self.update_count = update_count

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
#        current = self.get_monitor_value(logs)
#        if current is None:
#            return
#
#        if self.monitor_op(current, self.best):
#            self.best = current
#        else:
#            geom_val = self.geom_drop_layer.get_geom_val()+1
#            self.geom_drop_layer.set_geom_val(geom_val)
#            
#            stop_idx = self.stop_grad_layer.get_stop_idx()+1
#            self.stop_grad_layer.set_stop_idx(stop_idx)
        self.count+=1
        if self.count >self.update_count:
            self.bern_layer.set_temp(self.bern_layer.init_temp)
            
            geom_val = self.geom_drop_layer.get_geom_val()+1
            self.geom_drop_layer.set_geom_val(geom_val)
            
            stop_idx = self.stop_grad_layer.get_stop_idx()+1
            self.stop_grad_layer.set_stop_idx(stop_idx)
            
            self.count = 0
        else:
            temp = self.bern_layer.get_temp()*0.7
            self.bern_layer.set_temp(temp)
            

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




    
class BernoulliSampling(Layer):

    def __init__(self,init_temp,from_logits=False,**kwargs):
        super(BernoulliSampling, self).__init__(**kwargs)
        self.supports_masking = True
        self.init_temp = init_temp
        self.init = initializers.Constant(init_temp)
        self.from_logits = from_logits
    
    def set_temp(self,temperature):
        print('setting temperature: ', temperature)
        self.set_weights([np.asarray(temperature)])
    
    def get_temp(self):
        return self.get_weights()[0]
    
    def build(self,input_shape):
        self.temperature = self.add_weight(name="temperature",
                                     shape=(),
                                     dtype=K.floatx(),
                                     initializer=self.init,
                                     trainable=False)

    def call(self, inputs, training=None):
        _,latent_size = inputs.shape.as_list()
        def sampled():
            if self.from_logits:
                dist = tfp.distributions.RelaxedBernoulli(logits=inputs,
                                                          temperature=self.temperature)
            else:
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
        config = {'init_temp':self.init_temp,'from_logits':self.from_logits}
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
        self.valid_mask = None
        
    def set_geom_val(self,geom_val):
        self.geom_val = geom_val
        self.set_geom_indices()
        
    def get_geom_val(self):
        return self.geom_val
    
    def set_geom_indices(self):
        print('updating geom dropout indices, now at :',self.geom_val)
        _indices = np.expand_dims(np.arange(0,self.latent_size)-self.geom_val,axis=0)
        valid_mask = np.zeros(((1,self.latent_size,)))
        valid_mask[...,:self.geom_val+1] = 1
        self.set_weights([_indices,valid_mask])
        
    def build(self,input_shape):
        self.latent_size = input_shape.as_list()[1]
        self.indices = self.add_weight(name="geom_indices",
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
                

                return inputs * drop
            
            def valid_masked_only():
                valid_mask = tf.broadcast_to(self.valid_mask,K.shape(inputs))
                return inputs*valid_mask
            
            return K.in_train_phase(noised, valid_masked_only, training=training)
        return inputs

    def get_config(self):
        config = {'rate': self.rate,'geom_val':self.geom_val}
        base_config = super(GeometricDropout, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape
    
class StopGradientMask(Add):
    def __init__(self, stop_idx=0,**kwargs):
        super(StopGradientMask, self).__init__(**kwargs)
        self.stop_idx = stop_idx
        self.stop_gradient_mask = None
        self.gradient_mask = None
    
    def set_stop_idx(self,stop_idx):
        self.stop_idx = stop_idx
        self.set_masks()
        
    def get_stop_idx(self):
        return self.stop_idx
    
    def set_masks(self):
        print('setting grad mask : ', self.stop_idx)
        stop_gradient_mask = np.zeros(((1,self.latent_size,)))
        stop_gradient_mask[...,:self.stop_idx] = 1
        gradient_mask = np.ones(((1,self.latent_size,)))
        gradient_mask[...,:self.stop_idx] = 0

        self.set_weights([stop_gradient_mask,gradient_mask])
        
    def build(self,input_shape):
        super(StopGradientMask, self).build(input_shape)
        assert len(input_shape) == 2
        self.latent_size = input_shape[0].as_list()[1]

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

        self.set_masks()
    
    def _merge_function(self, inputs):
        assert len(inputs) == 2
        stopped_input,grad_input = inputs
        
        stop_gradient_mask = tf.broadcast_to(self.stop_gradient_mask,
                                             K.shape(stopped_input))
        gradient_mask= tf.broadcast_to(self.gradient_mask,
                                       K.shape(grad_input))
        
        return (tf.stop_gradient(stopped_input*stop_gradient_mask) 
                + grad_input*gradient_mask)

        
    def get_config(self):
        config = {'stop_idx': self.stop_idx}
        base_config = super(StopGradientMask, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        
class Quantizer(Layer):
    def __init__(self, **kwargs):
        super(Quantizer, self).__init__(**kwargs)
        
    def call(self, inputs):
        differentiable_round = tf.maximum(inputs-0.499,0)
        differentiable_round = differentiable_round * 10000
        differentiable_round = tf.minimum(differentiable_round, 1)
        return differentiable_round

        

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

def build_geosamp_block(input_shape,geom_rate,
                       init_temp=0.5,use_grad_stop_mask=True,
                       sampling=True,dropout=True,
                       from_logits=False):
    
    input_layer = keras.layers.Input(shape=input_shape,)
    x = input_layer
    
    if sampling:
        bern_layer = BernoulliSampling(init_temp=init_temp,
                                       from_logits=from_logits,
                                       name='bern_sampler')
        x_sampled = bern_layer(x)
        x_discrete = Quantizer()(x)
        x = StopGradientMask(name='stop_grad_mask')([x_discrete,x_sampled])
        
    tanh = Lambda(lambda x : (x-0.5)*2)(x)
    
    if dropout:
        out = GeometricDropout(geom_rate,name='geom_dropout')(tanh)
    else:
        out = tanh
        
    return keras.models.Model([input_layer],[out],name='geosampler')


def build_repeat_block(input_shape,num_repeats):
    input_layer = keras.layers.Input(shape=input_shape,name='repeat_input')
    
    out = RepeatBatch(num_repeats)(input_layer)

    return keras.models.Model([input_layer],[out],name='repeat_block')

def build_dropout_block(input_shape):
    input_layer = keras.layers.Input(shape=input_shape,name='dropout_input')
    out = Dropout(0.5)(input_layer)

    return keras.models.Model([input_layer],[out],name='dropout_block')

def build_latent_params(input_shape,latent_size,activation):
    input_layer = keras.layers.Input(shape=input_shape,name='latent_params_input')
    x = Dense(latent_size,activation=activation,name='latent_params_dense')(input_layer)
    return keras.models.Model([input_layer],[x],name='latent_params')

def build_sig_to_tanh_converter(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)
    tanh = Lambda(lambda x : (x-0.5)*2)(input_layer)
    return keras.models.Model([input_layer],[tanh],name='sig_to_tanh_converter')

def build_lin_to_tanh_converter(input_shape):
    input_layer = keras.layers.Input(shape=input_shape)
    tanh = Activation('tanh')(input_layer)
    return keras.models.Model([input_layer],[tanh],name='lin_to_tanh_converter')
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