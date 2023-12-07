import cvae_general
import pandas as pd
from tensorflow.keras.optimizers import Adam
import itertools
import helper
import argparse
import os

if __name__ == '__main__':
    ns = [8, 9, 10, 11]
    activation_functions = ['relu', 'sigmoid', 'tanh']
    latent_space_dimensions = [2, 4, 8]
    learning_rates = [0.001, 0.01]
    epsilons = [1e-7, 1e-5, 1e-3, 1e-1]
    layout_index = [0, 1, 2, 3, 4]

    lists = [ns, layout_index, activation_functions, latent_space_dimensions, learning_rates, epsilons]
    lists = list(itertools.product(*lists))

    lists = [x for x in lists if not (x[2] == 'relu' and x[4] == 0.01)]
    lists = [x for x in lists if not (x[2] == 'sigmoid' and (x[0] in [10, 11] or x[1] in [3, 4]))]
    lists = [x for x in lists if not (x[0] == 11 and x[1] == 4)]

    parser = argparse.ArgumentParser(prog="Model Training Script", description="Trains a CVAE model based on the index of the hyperparameter combination from the grid search and saves the model to a specified location")
    parser.add_argument('training_data', help="Path to .pickle, .csv, or .tsv for individual training samples", type=str)
    parser.add_argument('t_start', help="Position of first one-hot-encoded tissue in the training data", type=int)
    parser.add_argument('t_end', help="Position of last one-hot-encoded tissue in the training data", type=int)
    parser.add_argument('s_start', help="Position of first one-hot-encoded species in the training data", type=int)
    parser.add_argument('s_end', help="Position of last one-hot-encoded species in the training data", type=int)
    parser.add_argument('d_start', help="Position of first probe in the training data", type=int)
    parser.add_argument('index', help="Index of the hyperparameter combination", type=int)
    parser.add_argument('encoder_save_loc', help='Path to location to save trained encoder model', type=str)
    parser.add_argument('decoder_save_loc', help='Path to location to save trained decoder model', type=str)

    parser.add_argument('--val_seed', help="Random seed for selecting the validation dataset", default=-1, type=int)
    parser.add_argument('--seed', help="Random seed used to initiate model training", default=42, type=int)

    args = parser.parse_args()

    if os.path.splitext(args.training_data)[1] == '.pickle':
        training = pd.read_pickle(args.training_data)
    elif os.path.splitext(args.training_data)[1] == '.csv' or args.training_data.split('.', 1)[1] == 'csv.gz':
        training = pd.read_table(args.training_data, sep=',', index_col=0)
    else:
        training = pd.read_table(args.training_data, index_col=0)
    training = training.dropna(axis=1)
    print('Training data dimensions: ' + str(training.shape))

    tissue_index = training.columns.values[args.t_start:args.t_end+1]
    species_index = training.columns.values[args.s_start:args.s_end+1]

    train_data, val_data = helper.get_training_val_datasets(training, tissue_index, args.t_start, args.t_end, species_index, args.s_start, args.s_end, args.val_seed)

    Xtrain = train_data[train_data.columns[args.d_start:]]
    ytrain = train_data[train_data.columns[:args.d_start]]
    Xval = val_data[val_data.columns[args.d_start:]]
    yval = val_data[val_data.columns[:args.d_start]]

    print('Training and validation data dimensions')
    print(Xtrain.shape)
    print(ytrain.shape)
    print(Xval.shape)
    print(yval.shape)

    n = lists[args.index][0]
    layouts = [(1, [2**n]), (2, [2**n, 2**n]), (3, [2**n, 2**n, 2**n]), (2, [2**(n+1), 2**n]), (3, [2**(n+2), 2**(n+1), 2**n])]

    layout = layouts[lists[args.index][1]]
    activation_function = lists[args.index][2]
    latent_space = lists[args.index][3]
    learning_rate = lists[args.index][4]
    epsilon = lists[args.index][5]

    print('Hyperparameters')
    print(layout)
    print(activation_function)
    print(latent_space)
    print(learning_rate)
    print(epsilon)

    cvae, encoder, decoder = cvae_general.define_cvae(Xtrain, ytrain, latent_space, layout[0], layout[1], activation_function, args.seed)
    trained_cvae = cvae_general.train_cvae(cvae, Xtrain, ytrain, Xval, yval, 32, 50, Adam(learning_rate=learning_rate, epsilon=epsilon), 5)

    encoder.save(args.encoder_save_loc)
    decoder.save(args.decoder_save_loc)
