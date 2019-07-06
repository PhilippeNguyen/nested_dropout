import mnist
import argparse
import tensorflow as tf
import tensorflow.keras as keras
from special import (tanh_crossentropy,build_latent_block,
                     UpdateGeomRate,build_repeat_block)


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
        default=10,type=int,
        help="Early stopping patience"
    )
    parser.add_argument(
        "--batch_repeats", action="store", dest="batch_repeats",
        default=20,type=int,
        help="batch_repeats (total batch size = batch_size*batch_repeats)"
    )
    
    args = parser.parse_args()
    dataset_name = args.dataset
    latent_size = args.latent_size
    batch_size = args.batch_size
    geom_rate = args.geom_rate
    patience= args.patience
    epochs = args.epochs
    batch_repeats = args.batch_repeats
    out = args.save_model if args.save_model.endswith('.hdf5') else args.save_model + '.hdf5'
    if dataset_name == 'mnist':
        dataset = mnist
        if latent_size is None:
            latent_size = 100
    else:
        pass
    
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
    encoder_out = dataset.build_encoder(data_shape,latent_size)(input_layer)
    latent_shape = encoder_out.shape.as_list()[1:]
    
#    repeat_block = build_repeat_block(latent_shape,batch_repeats)
#    repeat_out = repeat_block(encoder_out)
    
    latent_block = build_latent_block(latent_shape,geom_rate=geom_rate,
                                      num_repeats=batch_repeats)
    latent_out = latent_block(encoder_out)
    
    decoder = dataset.build_decoder(latent_shape)(latent_out)
    
    model = keras.models.Model([input_layer],
                                [decoder])
    
    #start training
    model.add_loss(tanh_crossentropy(input_layer,decoder,
                                     batch_repeats=batch_repeats))
    
    model.compile(
            optimizer=keras.optimizers.Adam(),
            )
    
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    
    geom_layer = latent_block.get_layer('geom_dropout')
    update_geom = UpdateGeomRate(geom_layer)
    
    model.fit(x_train,validation_data=(x_test,None),
              batch_size=batch_size,
              epochs=epochs,
              callbacks=[early_stopping,update_geom],
              )
    model.save(out)
