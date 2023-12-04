import cvae_general
import pandas as pd
from tensorflow.keras.optimizers import Adam
import itertools
import numpy as np
import helper
import random
import os
import argparse

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

    print('Loading arguments')

    parser = argparse.ArgumentParser(prog="Hyperparameter Tuning Script", description="Trains a CVAE model based on a hyperparameter combination and saves the performance to a file")
    parser.add_argument('combo_averages', help="Path to .pickle, .csv, or .tsv for observed combination mean samples", type=str)
    parser.add_argument('training_data', help="Path to .pickle, .csv, or .tsv for individual training samples", type=str)
    parser.add_argument('t_start', help="Position of first one-hot-encoded tissue in the training data", type=int)
    parser.add_argument('t_end', help="Position of last one-hot-encoded tissue in the training data", type=int)
    parser.add_argument('s_start', help="Position of first one-hot-encoded species in the training data", type=int)
    parser.add_argument('s_end', help="Position of last one-hot-encoded species in the training data", type=int)
    parser.add_argument('d_start', help="Position of first probe in the training data", type=int)
    parser.add_argument('index', help="Index of the hyperparameter combination", type=int)
    parser.add_argument('output_dir', help="Path to output where hyperparameter combination performances will be scored", type=str)
    parser.add_argument('--val_seed', help="Random seed for selecting the validation dataset", default=-1, type=int)

    args = parser.parse_args()

    print(args.val_seed)

    if os.path.splitext(args.combo_averages)[1] == '.pickle':
        combo_averages = pd.read_pickle(args.combo_averages)
    elif os.path.splitext(args.combo_averages)[1] == '.csv':
        combo_averages = pd.read_table(args.combo_averages, sep=',', index_col=0)
    else:
        combo_averages = pd.read_table(args.combo_averages, index_col=0)

    if os.path.splitext(args.training_data)[1] == '.pickle':
        training = pd.read_pickle(args.training_data)
    elif os.path.splitext(args.training_data, sep=',', index_col=0)
    else:
        training = pd.read_table(args.training_data, index_col=0)
    training = training.dropna(axis=1)
    print('Training data dimensions: '+ str(training.shape))

    if not os.path.isdir(args.output_dir):
        os.mkdir(args.output_dir)
    if 'best' + str(args.index+1) + '.txt' in os.listdir(args.output_dir):
        print('Hyperparameter combination already trained')
        exit()

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

    print(layout)
    print(activation_function)
    print(latent_space)
    print(learning_rate)
    print(epsilon)

    max_performance = -1
    max_model = None

    rand_seed = random.randint(1, 10000)

    cvae, encoder, decoder = cvae_general.define_cvae(Xtrain, ytrain, latent_space, layout[0], layout[1], activation_function, rand_seed)
    trained_cvae = cvae_general.train_cvae(cvae, Xtrain, ytrain, Xval, yval, 32, 50, Adam(learning_rate=learning_rate, epsilon=epsilon), 5)

    if not np.isnan(trained_cvae.history['loss'][-1]):
        predictions = helper.predict_group_mean_normal(val_data, tissue_index, args.t_start, args.t_end+1, species_index, args.s_start, args.s_end+1, args.d_start, decoder, latent_space)
        result = helper.combo_mean_samplewise_performance_pearson(predictions, combo_averages)
        print(result)

        if result > max_performance:
            max_performance = result
            max_model = [layout, activation_function, latent_space, learning_rate, epsilon, rand_seed]

    print(args.index+1)
    print(max_model)
    print(max_performance)

    output_file = open(args.output_dir + '/best' + str(args.index+1) + '.txt', 'w')
    output_file.write(str(max_performance) + '\n')
    if max_model is not None:
        output_file.write(str(max_model[0][0]) + '\n')
        output_file.write(str(max_model[0][1]) + '\n')
        for i in range(1, 6):
            output_file.write(str(max_model[i]) + '\n')

    output_file.close()
