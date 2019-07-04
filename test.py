import mnist
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from special import tanh_crossentropy


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--folder", action="store", dest="folder",
        help="path to the folder of wavs"
    )
    parser.add_argument(
        "--latent_size", action="store", dest="latent_size",
        default=None,type=int,
        help="path to the folder of wavs"
    )
    parser.add_argument(
        "--dataset", action="store", dest="dataset",
        default='mnist',
        help="dataset (mnist or imagenet?)"
    )
    parser.add_argument(
        "--batch_size", action="store", dest="batch_size",
        default=32,type=int,
        help="batch_size"
    )
    parser.add_argument(
        "--save_model", action="store", dest="save_model",
        required=True,
        help="filename to save the model as (.hdf5)"
    )
    args = parser.parse_args()
    dataset_name = args.dataset
    latent_size = args.latent_size
    batch_size = args.batch_size
    out = args.save_model if args.save_model.endswith('.hdf5') else args.save_model + '.hdf5'
    if dataset_name == 'mnist':
        dataset = mnist
        if latent_size is None:
            latent_size = 100
    else:
        pass
    
    x_train,x_test,_,_ = dataset.get_data()
    data_shape = x_test.shape[1:]
    train_samples,test_samples = x_train.shape[0],x_test.shape[0]
    if (train_samples%batch_size) !=0:
        train_samples = train_samples-(train_samples%batch_size)
        x_train = x_train[:train_samples]
    if (test_samples%batch_size) !=0:
        test_samples = test_samples-(test_samples%batch_size)    
        x_test = x_test[:test_samples]
    
    input_layer = keras.layers.Input(shape=data_shape)
    encoder = dataset.build_encoder(data_shape,latent_size)(input_layer)
    latent_shape = encoder.shape.as_list()[1:]
    decoder = dataset.build_decoder(latent_shape)(encoder)
    
    model = keras.models.Model([input_layer],
                                [decoder])
    model.add_loss(tanh_crossentropy(input_layer,decoder))
    
    model.compile(
            optimizer=keras.optimizers.Adam(),
            )
    
    early_stopping = keras.callbacks.EarlyStopping(patience=10)
    model.fit(x_train,validation_data=(x_test,None),
              batch_size=batch_size,
              epochs=5,
              callbacks=[early_stopping]
              )
    model.save(out)