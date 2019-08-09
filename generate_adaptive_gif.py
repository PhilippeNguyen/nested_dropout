
import argparse
import tensorflow as tf
import tensorflow.keras as keras
import mnist
import special
import tensorflow.keras.backend as K
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation
import os
from os.path import join as pjoin



#%%
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--model", action="store", dest="model",
        help="path to the saved model"
    )
    parser.add_argument(
        "--output_folder", action="store", dest="output_folder",
        help="path to the output folder"
    )
    parser.add_argument(
        "--num_bits", action="store", dest="num_bits",
        type=int,default=25,
        help="Number of bits to generate"
    )
    parser.add_argument(
        "--num_images", action="store", dest="num_images",
        type=int,default=24,
        help="Number of images to generate (uses the first num_images of the test set)"
    )
    args = parser.parse_args()
    output_f = args.output_folder
    os.makedirs(output_f,exist_ok=True)
    num_bits = args.num_bits
    num_images = 32
    full_model = keras.models.load_model(args.model,compile=False,
                                         custom_objects=special.custom_objs)

    encoder = full_model.get_layer('encoder')
    latent = full_model.get_layer('latent_params')
    latent_sampler = full_model.get_layer('geosampler')
    decoder = full_model.get_layer('decoder')
    
    x_train,x_test,_,_ = mnist.get_data()
    input_data = x_test[:num_images]
    encoder_out = encoder.predict(input_data)
    latent_params = latent.predict(encoder_out)
    latent_out = latent_sampler.predict(latent_params)
#    decoded_latent = decoder.predict(latent_out)
    
    
    def predict_partial(num_bits):
        masked_vals = latent_out.copy()
        masked_vals[:,num_bits:] = 0
        return decoder.predict(masked_vals)

    results_list = [] 
    for bit_num in range(1,num_bits+1):
        results_list.append(predict_partial(bit_num)[...,0])
    
    for image_num in range(num_images):
        fig, ax = plt.subplots()
        
        ims = []
        for frame_idx,frame in enumerate(results_list):
            im = plt.imshow(frame[image_num], animated=True)
            title = latent_out[image_num,:frame_idx+1].copy()
            title[title==-1] = 0
            title = ''.join([str(x) for x in np.uint8(title).tolist()])
            title = ax.text(0.5,1.05,title, 
                size=plt.rcParams["axes.titlesize"],
                ha="center", transform=ax.transAxes, )
            ims.append([im,title])
        ani = animation.ArtistAnimation(fig, ims, interval=500, blit=False,
                                repeat_delay=1000)
        
        ani.save(pjoin(output_f,'idx_'+str(image_num)+'.gif'),writer='imagemagick')
        plt.close()