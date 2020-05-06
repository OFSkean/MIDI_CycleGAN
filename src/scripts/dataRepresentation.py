from music21 import *
import math
from scripts.util import *
import multiprocessing as mp

def representation_to_midi(encoded_midis, label):
    for idx, m in enumerate(encoded_midis):
        file_name = EXAMPLE_LOCATION_PATH + "4_" + label + ".midi"
        s = stream.Stream()

        # keeps track of which notes are on and when they started
        activeNotes = dict()
        for i in range(NUMBER_OF_SAMPLES):

            for j in range(NUM_OCTAVES):
                for k in range(NOTES_PER_OCTAVE):
                    ps = NUM_OCTAVES * j + k + 21

                    # look for new notes
                    if m[i, j, k] and (ps not in activeNotes):
                        # add note to our set if it's on
                        activeNotes[ps] = i

                    # look for notes that have just turned off
                    elif (not m[i, j, k]) and (ps in activeNotes):
                        n = note.Note()
                        noteStart = activeNotes.pop(ps)
                        n.offset = int(noteStart / SAMPLES_PER_BEAT)
                        n.quarterLength = (i - noteStart) / SAMPLES_PER_BEAT

                        # add note message to the stream
                        s.append(n)

        mf = s.write('midi', fp=file_name)


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
