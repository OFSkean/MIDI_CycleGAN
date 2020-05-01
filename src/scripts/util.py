import os
import numpy as np
import glob
from scripts.conf import *

"""
Gets all data filenames so they can be lazy loaded later in the program
"""
def load_dataset_filenames(A="jazz", B="classical"):
    # define what classes will be
    classA = GENRES[A]
    classB = GENRES[B]

    # load dataset filenames for each class
    datasetFilenamesA = glob.glob(os.path.join(DATA_PATH, classA, "formatted_*"))
    datasetFilenamesB = glob.glob(os.path.join(DATA_PATH, classB, "formatted_*"))

    return datasetFilenamesA, datasetFilenamesB


# select a batch of random samples, returns midis and target
def generate_real_samples(datasetFilenames, n_samples):
    # choose random samples
    ix = np.random.randint(0, datasetFilenames.shape[0], n_samples)
    X = np.ones((n_samples, NUMBER_OF_SAMPLES, NUM_OCTAVES, NOTES_PER_OCTAVE))

    # load samples from files
    for midiFile in range(ix.shape[0]):
        X[midiFile, :] = np.load(datasetFilenames[midiFile]).reshape((NUMBER_OF_SAMPLES, NOTES_PER_OCTAVE, NOTES_PER_OCTAVE, 1))

    # generate 'real' class labels (1)
    y = np.ones((n_samples, 1, 1, 1, 1))
    return X, y


# generate a batch of midis, returns midis and target
def generate_fake_samples(g_model, dataset):
    # generate fake instance
    X = np.clip(np.round(g_model.predict(dataset)), a_min=0, a_max=1)

    # create 'fake' class labels (0)
    y = np.zeros((len(X), 1, 1, 1, 1))

    return X, y


def update_midi_pool(pool, midis, max_size=50):
    selected = list()
    for midi in midis:
        # stock the pool if it's not full
        if len(pool) < max_size:
            pool.append(midi)
            selected.append(midi)

        # use image, but don't add it to the pool
        elif np.random.random() < POOL_UPDATE_CHANCE:
            selected.append(midi)

        # replace an existing image and use replaced image
        else:
            ix = np.random.randint(0, len(midis))
            selected.append(pool[ix])
            pool[ix] = midi

    return np.asarray(selected)
