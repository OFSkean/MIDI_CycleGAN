"""
This file sets the configuration and constants used throughout the project
"""

"""
Constants
"""
NOTES_PER_OCTAVE = 12


"""
Configuration
"""
# DATA CONFIGURATION
NUM_OCTAVES = 9
SAMPLES_PER_BEAT = 16 * 3
NUMBER_OF_BEATS_TO_CAPTURE = 300
NUMBER_OF_SAMPLES = SAMPLES_PER_BEAT * NUMBER_OF_BEATS_TO_CAPTURE
GENRES = ["jazz", "classical", "pop"]

# PATHS
EXAMPLE_LOCATION_PATH = "H:/sandbox/datasets/midi/examples/"
DATA_PATH = "H:/sandbox/datasets/midi/music"

# LEARNING CONFIG
midi_shape = (NUMBER_OF_BEATS_TO_CAPTURE * SAMPLES_PER_BEAT, NUM_OCTAVES, NOTES_PER_OCTAVE, 1)
n_samples = 10
poolSize = 50
n_epochs = 10
n_batch = 1
save_interval = 20
POOL_UPDATE_CHANCE = 0.5

# checkpoint config
checkpoint_path = "H:/sandbox/datasets/midi/models/"

# program config
trainModels = False
