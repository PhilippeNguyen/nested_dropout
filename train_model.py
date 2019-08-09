
import mnist
import argparse
import tensorflow as tf
import json
import tensorflow.keras as keras
#
#import keras

from special import (tanh_crossentropy,build_geosamp_block,
                     UpdateExtraParams,build_repeat_block,FixedModelCheckpoint,
                     build_dropout_block,custom_objs,TrainSequence,
                     build_latent_params)
from tensorflow.keras.callbacks import ModelCheckpoint

#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
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
        "--pretrain_model", action="store", dest="pretrain_model",
        default=None,
        help="pretrain_model from pretrain_model.py"
    )
    parser.add_argument(
        "--save_model", action="store", dest="save_model",
        required=True,
        help="filename to save the model as (.hdf5)"
    )
    parser.add_argument(
        "--config_json", action="store", dest="config_json",
        default=None,
        help="json file with dict, variables in this json will replace variables in this script"
    )
    
    args = parser.parse_args()
    dataset_name = args.dataset
    pretrain_model_path = args.pretrain_model
    out = args.save_model if args.save_model.endswith('.hdf5') else args.save_model + '.hdf5'
    if dataset_name == 'mnist':
        dataset = mnist
        
    config_json = args.config_json

    
    ###Config Stuff

    #Change
    batch_size = 8
    batch_repeats = 200
    save_best_only =False
    latent_size = 100
    lr= 0.001
    patience = 10
    geom_rate = 0.95
    epochs = 200
    update_count = 4
    init_temp=0.5
    
    #Don't change
    use_grad_stop_mask=True
    sampling=True
    dropout=True
    
    if config_json is not None:
        config_json = json.load(open(config_json,'r'))
        locals().update(config_json)
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


    pretrain_model = keras.models.load_model(pretrain_model_path,
                                             custom_objects=custom_objs)
    encoder = pretrain_model.get_layer('encoder')
    encoder.trainable = False
    latent_params = pretrain_model.get_layer('latent_params')
    decoder = pretrain_model.get_layer('decoder')
    
    
    input_layer = keras.layers.Input(shape=data_shape)
    
    encoder_out = encoder(input_layer)
    encoder_out_shape = encoder_out.shape.as_list()[1:]
    
    latent_out = latent_params(encoder_out)
    latent_out_shape = latent_out.shape.as_list()[1:]
    
    repeat_block = build_repeat_block(latent_out_shape,batch_repeats)
    repeat_out = repeat_block(latent_out)
    repeat_out_shape = repeat_out.shape.as_list()[1:]
    
    geosamp_block = build_geosamp_block(repeat_out_shape,
                                      geom_rate=geom_rate,
                                      init_temp=init_temp,
                                      use_grad_stop_mask=use_grad_stop_mask,
                                      sampling=sampling,
                                      dropout=dropout,
                                      from_logits=True,
                                      )
    geosamp_out = geosamp_block(repeat_out)
    
    decoder_out = decoder(geosamp_out)
        
        
    repeat_block = build_repeat_block(data_shape,batch_repeats)
    repeat_input = repeat_block(input_layer)
    
    training_model = keras.models.Model([input_layer],
                                   [decoder_out])
    
    early_stopping = keras.callbacks.EarlyStopping(patience=patience)
    geom_layer = geosamp_block.get_layer('geom_dropout')
    stop_grad_layer = geosamp_block.get_layer('stop_grad_mask')
    bern_layer = geosamp_block.get_layer('bern_sampler')
    update_extra= UpdateExtraParams(geom_drop_layer=geom_layer,
                                 stop_grad_layer=stop_grad_layer,
                                 bern_layer=bern_layer,
                                 update_count=update_count)
    

    saving_model = keras.models.Model([input_layer],
                                   [decoder_out])
    x_train_seq = TrainSequence(x_train,None,batch_size=batch_size)
    x_test_seq = TrainSequence(x_test,None,batch_size=batch_size)
    training_model.add_loss(tanh_crossentropy(repeat_input,decoder_out))

    training_model.compile(
        optimizer=keras.optimizers.Adam(lr=lr),
        )
    model_check = FixedModelCheckpoint(out,saving_model,save_best_only=save_best_only)
    training_model.fit_generator(x_train_seq,
                                 validation_data=x_test_seq,
                                  epochs=epochs,
                                  callbacks=[early_stopping,update_extra,model_check],
                                  steps_per_epoch=200,
                                  validation_steps=200,
                                  )

