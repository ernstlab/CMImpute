import helper
import pandas as pd
from tensorflow import keras
import argparse
import cvae_general
import os

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='Impute Combination Mean Samples', description='Uses a trained decoder to impute species-tissue combination mean samples')
    parser.add_argument('data', help='Path to .pickle, .csv, or .tsv with either testing or training data (used to one-hot-encoded label ordering and to extract combinations to be imputed if testing)', type=str)
    parser.add_argument('t_start', help='Position of first one-hot-encoded tissue in the training data', type=int)
    parser.add_argument('t_end', help='Position of last one-hot-encoded tissue in the training data', type=int)
    parser.add_argument('s_start', help='Position of first one-hot-encoded species in the training data', type=int)
    parser.add_argument('s_end', help='Position of last one-hot-encoded species in the training data', type=int)
    parser.add_argument('d_start', help='Position of first probe in the training data', type=int)
    parser.add_argument('latent_space_dimension', help='Latent space dimension needed for input into the decoder', type=int)
    parser.add_argument('decoder', help='Path to .model file for the trained decoder', type=str)
    parser.add_argument('pred_save_loc', help='Path to output where predictions will be saved', type=str)

    parser.add_argument('--tissue', help='Tissue of single species-tissue combination to impute', type=str, default='No tissue provided')
    parser.add_argument('--species', help='Species of single species-tissue combination to impute', type=str, default='No species provided')
    parser.add_argument('--input_file', help='Tab-delimited file containing species-tissue combinations to impute', type=str, default='No file provided')

    args = parser.parse_args()

    if os.path.splitext(args.data)[1] == '.pickle':
        data = pd.read_pickle(args.data)
    elif os.path.splitext(args.data)[1] == '.csv' or args.data.split('.', 1)[1] == 'csv.gz':
        data = pd.read_table(args.data, sep=',', index_col=0)
    else:
        data = pd.read_table(args.data, index_col=0)
    data = data.dropna(axis=1)

    decoder = keras.models.load_model(args.decoder)
    decoder.compile(loss='mse')

    tissue_index = data.columns.values[args.t_start:args.t_end+1]
    species_index = data.columns.values[args.s_start:args.s_end+1]

    if (args.tissue == 'No tissue provided' or args.species == 'No species provided') and args.input_file == 'No file provided':
        print('Imputing combinations present in testing dataset')
        predictions = helper.predict_group_mean_normal(data, tissue_index, args.t_start, args.t_end+1, species_index, args.s_start, args.s_end+1, args.d_start, decoder, args.latent_space_dimension)
    elif args.tissue != 'No tissue provided' and args.species != 'No species provided' and args.input_file == 'No file provided':
        print('Imputing (' + args.tissue + ', ' + args.species + ')')
        predictions = helper.predict_group_mean_normal(data, tissue_index, args.t_start, args.t_end+1, species_index, args.s_start, args.s_end+1, args.d_start, decoder, args.latent_space_dimension, testing=False, species=args.species, tissue=args.tissue)
    elif args.input_file != 'No file provided' and args.tissue == 'No tissue provided' and args.species == 'No species provided':
        print('Imputed combinations present in input file')
        predictions = helper.predict_group_mean_normal(data, tissue_index, args.t_start, args.t_end+1, species_index, args.s_start, args.s_end+1, args.d_start, decoder, args.latent_space_dimension, testing=False, input_file=args.input_file)
    else:
        print('Conflicting arguments provided')

    print('Number of combinations imputed: '+ str(len(predictions)))

    if os.path.splitext(args.pred_save_loc)[1] == '.pickle':
        pd.DataFrame.from_dict(predictions).transpose().to_pickle(args.pred_save_loc)
    elif os.path.splitext(args.pred_save_loc)[1] == '.csv':
        pd.DataFrame.from_dict(predictions).transpose().to_csv(args.pred_save_loc)
    else:
        pd.DataFrame.from_dict(predictions).transpose().to_csv(args.pred_save_loc, sep='\t')


