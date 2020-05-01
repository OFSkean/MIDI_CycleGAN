from music21 import *
import math
from scripts.util import *
import multiprocessing as mp

"""
# check that encoding and decoding are invertible:
def check_representation():
    a, _, b, _ = load_datasets()

    sizesA = [np.sum(i) for i in a]
    sizesB = [np.sum(i) for i in b]
    plt.hist(sizesB)
    plt.show()


def representation_to_midi(encoded_midis, label):
    for idx, m in enumerate(encoded_midis):
        file_name = EXAMPLE_LOCATION_PATH + "4_" + label.split('\\')[-1] + ".midi"

        s = stream.Stream()
        lastINote = 0
        for i in range(400):
            for j in range(8):
                for k in range(12):
                    if m[i, j, k]:
                        length, increment = 1, 1
                        n = note.Note(8 * j + k + 21)
                        n.quarterLength = length / 8
                        n.offset = i - lastINote
                        s.append(n)
                        lastINote = i
        print(len(s.notes))
        mf = s.write('midi', fp=file_name)
"""

"""
Given: The path to a MIDI file
Returns: A matrix that represents the notes from the MIDI file
"""
def midi_to_representation(file_location):
    notesMatrix = np.zeros(shape=(NUMBER_OF_SAMPLES, NOTES_PER_OCTAVE * NUM_OCTAVES))

    loadedFile = converter.parse(file_location)
    for n in loadedFile.recurse().notes:
        timeStamp = int(n.offset * SAMPLES_PER_BEAT)  # the index in our matrix that a note starts at
        duration = max(1, math.ceil(
            n.duration.quarterLength * SAMPLES_PER_BEAT))  # the duration of a note, in terms of quarter note lengths

        # if note ends before the cutoff, don't allow it
        # todo: make ALL notes that start before the cutoff permissible
        if (timeStamp + duration) < NUMBER_OF_SAMPLES:
            # add notes to the representation
            if isinstance(n, note.Note):
                notesMatrix[timeStamp:timeStamp + duration, int(n.pitch.ps - 21)] = n.volume.velocity

            # break down chords into individual notes and add them
            elif isinstance(n, chord.Chord):
                for nn in n:
                    notesMatrix[timeStamp:timeStamp + duration, int(nn.pitch.ps - 21)] = n.volume.velocity

    return notesMatrix.reshape((NUMBER_OF_SAMPLES, NUM_OCTAVES, NOTES_PER_OCTAVE))


"""
This was specifically designed as a modular function to be called in parallel. 
MIDI conversions are pretty slow sequentially.

Given: A MIDI location
Returns: None
Does: Saves a Numpy array of our representation
"""
def create_representations(file_location):
    sp = file_location.split('\\')
    ffn = '\\'.join(sp[:-1]) + "\\formatted_" + sp[-1][:-4] + ".npy"
    try:
        if not os.path.exists(ffn):
            np.save(ffn, midi_to_representation(file_location))
            print("created ", ffn)
    except Exception as e:
        print(e)


def multi_process_midis():
    midiFileNames = []
    for genre in GENRES:
        midiFileNames.extend(glob.glob(os.path.join(DATA_PATH, genre, "*.mid")))

    p = mp.Pool(mp.cpu_count())
    p.map(create_representations, midiFileNames)


if __name__ == "__main__":
    multi_process_midis()
