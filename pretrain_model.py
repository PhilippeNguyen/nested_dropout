
import mnist
import argparse
import tensorflow as tf

import tensorflow.keras as keras
#
#import keras

from special import (tanh_crossentropy,
                     UpdateGeomRate,build_repeat_block,FixedModelCheckpoint,
                     build_dropout_block,build_latent_params,build_tanh_converter)
from tensorflow.keras.callbacks import ModelCheckpoint

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--latent_size", action="store", dest="latent_size",
        default=None,type=int,
        help="number of neurons in the latent"
    )
    parser.add_argument(
        "--dataset", action="store", dest="dataset",
        default='mnist',
        help="dataset (mnist or imagenet?)"
    )
    parser.add_argument(
        "--batch_size", action="store", dest="batch_size",
        default=8,type=int,
        help="batch_size (total batch size = batch_size*batch_repeats)"
    )
    parser.add_argument(
        "--save_model", action="store", dest="save_model",
        required=True,
        help="filename to save the model as (.hdf5)"
    )
    parser.add_argument(
        "--geom_rate", action="store", dest="geom_rate",
        default=0.95,type=float,
        help="geometric dropout rate"
    )
    parser.add_argument(
        "--epochs", action="store", dest="epochs",
        default=100,type=int,
        help="number of epochs to train"
    )
    parser.add_argument(
        "--patience", action="store", dest="patience",
        default=5,type=int,
        help="Early stopping patience"
    )
    
    args = parser.parse_args()
    dataset_name = args.dataset
    latent_size = args.latent_size
    batch_size = args.batch_size
    geom_rate = args.geom_rate
    patience= args.patience
    epochs = args.epochs
    out = args.save_model if args.save_model.endswith('.hdf5') else args.save_model + '.hdf5'
    if dataset_name == 'mnist':
        dataset = mnist
        if latent_size is None:
            latent_size = 100
    else:
        pass
    
    ###Config Stuff

    #overwrite the parser args
    batch_size = 64
    ###
    
    #Set up data
    x_train,x_test,_,_ = dataset.get_data()
    data_shape = x_test.shape[1:]
    train_samples,test_samples = x_train.shape[0],x_test.shape[0]
    if (train_samples%batch_size) !=0:
        train_samples = train_samples-(train_samples%batch_size)
        x_train = x_train[:train_samples]
    if (test_samples%batch_size) !=0:
        test_samples = test_samples-(test_samples%batch_size)    
        x_test = x_test[:test_samples]
    
    #Set up model
    input_layer = keras.layers.Input(shape=data_shape)
    encoder = dataset.build_encoder(data_shape)
    encoder_out = encoder(input_layer)
    encoder_out_shape = encoder_out.shape.as_list()[1:]
    
    latent_param_block = build_latent_params(encoder_out_shape,
                                             latent_size=latent_size)
    latent_params_out = latent_param_block(encoder_out)
    latent_params_shape = latent_params_out.shape.as_list()[1:]
    
    tanh_converter = build_tanh_converter(latent_params_shape)
    tanh_out = tanh_converter(latent_params_out)
    
    drop_block = build_dropout_block(latent_params_shape)
    drop_out = drop_block(tanh_out)
    latent_shape = drop_out.shape.as_list()[1:]
    
    decoder = dataset.build_decoder(latent_shape)
    decoder_out = decoder(drop_out)
    
    pretrain_model = keras.models.Model([input_layer],
                                        [decoder_out])
    saving_model = keras.models.Model([input_layer],
                               [decoder_out])


    pretrain_model.compile(
            optimizer=keras.optimizers.Adam(),
            loss=tanh_crossentropy,
            )
#    #    start training
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    model_check = ModelCheckpoint(out,save_best_only=True)
    pretrain_model.fit(x_train,x_train,validation_data=(x_test,x_test),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping,model_check],
              )