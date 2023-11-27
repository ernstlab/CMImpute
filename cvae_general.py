import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
import warnings
import numpy as np
from tensorflow.keras.layers import Input, Dense, Lambda, Dropout
from tensorflow.keras.layers import concatenate as concat
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import random as python_random

warnings.filterwarnings('ignore')
tf.compat.v1.disable_eager_execution()

def sample_z(args):
    mu, l_sigma = args
    eps = K.random_normal(shape=(K.shape(mu)[0], K.shape(mu)[1]), mean=0., stddev=1., seed=42)
    return mu + K.exp(l_sigma / 2) * eps

def vae_loss(y_true, y_pred):
    recon = K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)
    kl = 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=-1)
    return recon + kl

def KL_loss(y_true, y_pred):
    return 0.5 * K.sum(K.exp(l_sigma) + K.square(mu) - 1. - l_sigma, axis=1)

def recon_loss(y_true, y_pred):
    return K.sum(K.binary_crossentropy(y_true, y_pred), axis=-1)

def define_cvae(X_train, y_train, latent_space_size, num_hidden_layers, hidden_layer_dims, activ, random_seed):
    os.environ['PYTHONHASHSEED'] = str(random_seed)
    np.random.seed(random_seed)
    python_random.seed(random_seed)
    tf.random.set_seed(random_seed)
    tf.compat.v1.set_random_seed(random_seed)

    os.environ['TF_DETERMINISTIC_OPS'] = str(1)
    os.environ['TF_CUDNN_DETERMINISTIC'] = str(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
    tf.config.threading.set_intra_op_parallelism_threads(1)

    n_x = X_train.shape[1]
    n_y = y_train.shape[1]

    X = Input(shape=(n_x,), name='x')
    label = Input(shape=(n_y,), name='labels')
    inputs = concat([X, label], name='cvae_inputs')

    global mu
    global l_sigma

    encoder_layers = []
    for i in range(num_hidden_layers):
        encoder_layers.append(Dense(hidden_layer_dims[i], activation=activ, name='encoder_layer_' + str(i)))

    encoder_layers[0] = encoder_layers[0](inputs)
    for i in range(1, num_hidden_layers):
        encoder_layers[i] = encoder_layers[i](encoder_layers[i - 1])

    mu = Dense(latent_space_size, activation='linear', name='mu')(encoder_layers[-1])
    l_sigma = Dense(latent_space_size, activation='linear', name='sigma')(encoder_layers[-1])

    z = Lambda(sample_z, output_shape=(latent_space_size,), name='z')([mu, l_sigma])
    zc = concat([z, label], name='zc')

    decoder_layers = []
    for i in range(num_hidden_layers):
        decoder_layers.append(Dense(hidden_layer_dims[num_hidden_layers - 1 - i], activation=activ, name='decoder_layer_' + str(i)))
    decoder_out = Dense(n_x, activation='sigmoid', name='output')

    cvae_trained_decoder = [decoder_layers[0](zc)]
    for i in range(1, num_hidden_layers):
        cvae_trained_decoder.append(decoder_layers[i](cvae_trained_decoder[i - 1]))

    outputs = decoder_out(cvae_trained_decoder[-1])

    cvae_to_train = Model([X, label], outputs)

    encoder_to_train = Model([X, label], mu)

    d_in = Input(shape=(latent_space_size + n_y,), name='decoder_input')
    decoder_trained = [decoder_layers[0](d_in)]
    for i in range(1, num_hidden_layers):
        decoder_trained.append(decoder_layers[i](decoder_trained[i - 1]))
    d_out = decoder_out(decoder_trained[-1])

    decoder_to_train = Model(d_in, d_out)

    return [cvae_to_train, encoder_to_train, decoder_to_train]

def train_cvae(cvae_to_train, X_train, y_train, X_test, y_test, batchsize, n_epoch, optim, pat):
    cvae_to_train.compile(optimizer=optim, loss=vae_loss, metrics=[KL_loss, recon_loss])
    return cvae_to_train.fit([X_train, y_train], X_train, verbose=1, batch_size=batchsize,
                             epochs=n_epoch, validation_data=([X_test, y_test], X_test),
                             callbacks=[EarlyStopping(patience=pat)])

def single_encoded_sample(x_row, y_row, trained_encoder):
    return trained_encoder.predict([x_row.reshape(1, x_row.shape[0]), y_row.reshape(1, y_row.shape[0])])

def single_decoded_sample(encoded_sample, trained_decoder):
    return trained_decoder.predict(encoded_sample)
