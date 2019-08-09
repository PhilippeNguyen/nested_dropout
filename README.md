# Nested Dropout
Implementation of Wavenet (https://arxiv.org/abs/1402.0915), but modernized.
Uses Gumbel-Bernoulli distribution for the latent vector.

Generates an ordered code for the latent space of the autoencoder, this has many uses e.g. adaptive compression.

#### Requirements
 * tensorflow
 * tensorflow_probability
 * numpy
 * matplotlib

#### Usage
First pretrain a model with the pretrain_model.py script (should only take a few minutes).
Then feed this pretrain model to the train_model.py script (this will take awhile).

#### TODO
* Add KL term so it's similar to a VAE
* code clean up
