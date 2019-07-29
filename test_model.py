import argparse
import tensorflow as tf
import tensorflow.keras as keras
import mnist
import special
import tensorflow.keras.backend as K
import numpy as np
#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", action="store", dest="model",
        help="path to the saved model"
    )
    args = parser.parse_args()
    full_model = keras.models.load_model(args.model,compile=False,
                                         custom_objects=special.custom_objs)
    encoder = full_model.get_layer('encoder')
    latent = full_model.get_layer('latent_params')
    latent_sampler = full_model.get_layer('geosampler')
#    latent_sampler = full_model.get_layer('geosamp_block')
    decoder = full_model.get_layer('decoder')
    
    x_train,x_test,_,_ = mnist.get_data()
    encoder_out = encoder.predict(x_test)
    latent_params = latent.predict(encoder_out)
    latent_out = latent_sampler.predict(latent_params)
    decoded_latent = decoder.predict(latent_out)
#    
    
#%%
#model_path = '/tmp/pretrain.hdf5'
#full_model = keras.models.load_model(model_path,compile=False,
#                                     custom_objects=special.custom_objs)
#encoder = full_model.get_layer('encoder')
#latent = full_model.get_layer('latent_params')
#
#
#x_train,x_test,_,_ = mnist.get_data()
#encoder_out = encoder.predict(x_test)
#latent_params = latent.predict(encoder_out)
