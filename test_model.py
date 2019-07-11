import argparse
import tensorflow as tf
import tensorflow.keras as keras
import mnist
import special
import tensorflow.keras.backend as K

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
    latent = full_model.get_layer('latent_block')
    decoder = full_model.get_layer('decoder')
    
    x_train,x_test,_,_ = mnist.get_data()
    latent_params = encoder.predict(x_test)
    latent_out = latent.predict(latent_params)
    decoded_latent = decoder.predict(latent_params)
#    
#    K.set_learning_phase(1)
#    
#    latent_shape = encoder.output.shape.as_list()[1:]
#    latent_block = special.build_latent_block(latent_shape,geom_rate=0.95)
#    geom_layer = latent_block.get_layer('geom_dropout')
#    out = latent_block.predict(latent_params)