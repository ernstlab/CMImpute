from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
import numpy as np
import cvae_general
import pandas as pd
import random

def get_training_val_datasets(training, tissue_index, t_start, t_end, species_index, s_start, s_end, train_val_split_random_seed):
    training_combos = get_combo_list(training, tissue_index, t_start, t_end + 1, species_index, s_start, s_end + 1)

    if train_val_split_random_seed == -1:
        print('Selecting random seed for training-testing data split')
        found_val_set = False
        iters = 0
        while not found_val_set:
            iters += 1
            rand = random.randint(1, 10000)

            train_combos, val_combos = get_single_train_val_combination_split(training_combos, rand)

            if len(val_combos) / (len(val_combos) + len(train_combos)) > 0.10:
                found_val_set = True
        print('Number of iterations: ' + str(iters))
        print('Random seed: ' + str(rand))
    else:
        train_combos, val_combos = get_single_train_val_combination_split(training_combos, train_val_split_random_seed)

    if len(val_combos) / (len(val_combos) + len(train_combos)) > 0.1:
        print('> 10% of combos used for validation')

    print('Number of training species-tissue combinations: ' + str(len(train_combos)))
    print('Number of validation species-tissue combinations: ' + str(len(val_combos)))
    print(len(val_combos) / (len(val_combos) + len(train_combos)))

    combo_individuals, _, _ = get_species_tissue_combos(training, tissue_index, t_start, t_end + 1, species_index, s_start, s_end + 1)

    train_data = []
    for combo in train_combos:
        train_data.append(training.loc[combo_individuals[combo]])
    train_data = pd.concat(train_data)

    val_data = []
    for combo in val_combos:
        val_data.append(training.loc[combo_individuals[combo]])
    val_data = pd.concat(val_data)

    return train_data, val_data

def get_single_train_val_combination_split(training_combos, random_seed):
    train_combos, val_combos = train_test_split(training_combos, test_size=0.2, random_state=random_seed)

    to_remove = []
    for val_combo in val_combos:
        same_spec = val_combo[1] in list(zip(*train_combos))[1]
        same_tiss = val_combo[0] in list(zip(*train_combos))[0]

        if not same_spec or not same_tiss:
            to_remove.append(val_combo)

    train_combos += to_remove
    val_combos = [x for x in val_combos if x not in to_remove]

    return train_combos, val_combos

def get_species_tissue_combos(data, tiss_index, tiss_start, tiss_end, spec_index, spec_start, spec_end):
    subject_tissue_species = {}
    for index, row in data.iterrows():
        cur_tiss = tiss_index[np.where(row.values[tiss_start:tiss_end] == 1)[0][0]]
        # tiss = cur_tiss[cur_tiss.index('_') + 1:]
        cur_spec = spec_index[np.where(row.values[spec_start:spec_end] == 1)[0][0]]
        # spec = cur_spec[cur_spec.index('_') + 1:]

        subject_tissue_species[index] = (cur_tiss, cur_spec)

    tissue_species_combos = {}
    tissue_combos = {}
    species_combos = {}
    for item in subject_tissue_species:
        t, s = subject_tissue_species[item]

        if t not in tissue_combos:
            tissue_combos[t] = [item]
        else:
            tissue_combos[t].append(item)

        if s not in species_combos:
            species_combos[s] = [item]
        else:
            species_combos[s].append(item)

        if (t, s) not in tissue_species_combos:
            tissue_species_combos[(t, s)] = [item]
        else:
            tissue_species_combos[(t, s)].append(item)

    return tissue_species_combos, tissue_combos, species_combos

def get_combo_list(training, tiss_index, tiss_start, tiss_end, spec_index, spec_start, spec_end):
    combos_present = []

    for _, row in training.iterrows():
        t_index = np.where(row.values[tiss_start:tiss_end] == 1)[0][0]
        s_index = np.where(row.values[spec_start:spec_end] == 1)[0][0]

        c = (tiss_index[t_index], spec_index[s_index])

        if c not in combos_present:
            combos_present.append(c)

    return combos_present

def predict_group_mean_normal(data, tiss_index, tiss_start, tiss_end, spec_index, spec_start, spec_end, data_start, decoder, latent_space_dim, testing=True, species=None, tissue=None, input_file=None):
    if testing:
        tissue_species_combos = get_combo_list(data, tiss_index, tiss_start, tiss_end, spec_index, spec_start, spec_end)
    elif species is not None and tissue is not None:
        tissue_species_combos = [('Tissue_' + tissue, 'Species_' + species)]
    elif species is None and tissue is None and input_file is not None:
        tissue_species_combos = pd.read_table(input_file, header=None).values
        tissue_species_combos = [('Tissue_' + x[0], 'Species_' + x[1]) for x in tissue_species_combos]

    combo_predictions = {}

    for combo in tissue_species_combos:
        print(combo)
        label = np.zeros(data_start)
        encoded = np.random.normal(size=(1, latent_space_dim))

        t_index = np.where(tiss_index == combo[0])[0][0]
        s_index = np.where(spec_index == combo[1])[0][0]
        label[tiss_start + t_index] = 1
        label[spec_start + s_index] = 1

        encoded = np.concatenate([encoded, label.reshape(1, label.shape[0])], axis=1)
        decoded = cvae_general.single_decoded_sample(encoded, decoder)
        decoded = decoded.reshape(decoded.shape[1])

        print(decoded.shape)

        combo_predictions[combo] = decoded

    return combo_predictions

def combo_mean_samplewise_performance_pearson(predictions, observed):
    corrs = []
    for combo in predictions:
        print(combo)
        obs = observed.loc[combo[0][combo[0].index('_')+1:], combo[1][combo[1].index('_')+1:]]
        corrs.append(pearsonr(predictions[combo], obs)[0])
        print(corrs[-1])

    return np.mean(corrs)
