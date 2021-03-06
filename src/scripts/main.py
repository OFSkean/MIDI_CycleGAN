from scripts.networks import *
from scripts.dataRepresentation import *
import time
import keras
import matplotlib.pyplot as plt

# configuration setup for checkpoints
checkpoint_dir = os.path.dirname(checkpoint_path)
# Create a callback that saves the model's weights
cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                 save_weights_only=True,
                                                 verbose=1,
                                                 period = 10)

# main train function
def train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoB, c_model_BtoA, datasetFilenames):
    # unpack dataset
    trainA, trainB = datasetFilenames
    # prepare midi pool for fakes
    poolA, poolB = [], []

    # calculate the number of training iterations
    n_steps = n_batch * n_epochs

    # manually enumerate epochs
    for i in range(n_steps):
        print("Starting iteration " + str(i+1))
        start = time.time()

        # select a batch of real samples
        X_realA, y_realA = generate_real_samples(trainA, n_samples)
        X_realB, y_realB = generate_real_samples(trainB, n_samples)

        # generate a batch of fake samples
        X_fakeA, y_fakeA = generate_fake_samples(g_model_BtoA, X_realB)
        X_fakeB, y_fakeB = generate_fake_samples(g_model_AtoB, X_realA)

        # update fakes from pool
        X_fakeA = update_midi_pool(poolA, X_fakeA)
        X_fakeB = update_midi_pool(poolB, X_fakeB)

        # update discriminator for A -> [real/fake]
        print("\tUpdating discriminator A")
        dA_loss1 = d_model_A.train_on_batch(X_realA, y_realA)
        dA_loss2 = d_model_A.train_on_batch(X_fakeA, y_fakeA)

        # update generator B->A via adversarial and cycle loss
        print("\tUpdating generator B->A")
        g_loss2, _, _, _, _ = c_model_BtoA.train_on_batch([X_realB, X_realA], [y_realA, X_realA, X_realB, X_realA])

        # update discriminator for B -> [real/fake]
        print("\tUpdating discriminator B")
        dB_loss1 = d_model_B.train_on_batch(X_realB, y_realB)
        dB_loss2 = d_model_B.train_on_batch(X_fakeB, y_fakeB)

        # update generator A->B via adversarial and cycle loss
        print("\tUpdating generator A->B")
        g_loss1, _, _, _, _ = c_model_AtoB.train_on_batch([X_realA, X_realB], [y_realB, X_realB, X_realA, X_realB])

        # summarize performance
        print('\tA[%.3f,%.3f] dB[%.3f,%.3f] g[%.3f,%.3f]' % (dA_loss1, dA_loss2, dB_loss1, dB_loss2, g_loss1, g_loss2))
        print('\tElapsed time: %.5f' % (time.time() - start))

        if not (i+1) % save_interval:
            print("\nSaving weights\n")
            d_model_A.save_weights(checkpoint_path + "dA_" + str(i+1))
            d_model_B.save_weights(checkpoint_path + "dB_" + str(i+1))
            g_model_AtoB.save_weights(checkpoint_path + "gAB_" + str(i+1))
            g_model_BtoA.save_weights(checkpoint_path + "gBA_" + str(i+1))
            c_model_AtoB.save_weights(checkpoint_path + "cAB_" + str(i+1))
            c_model_BtoA.save_weights(checkpoint_path + "cBA_" + str(i+1))


def main():
    # define generators
    g_model_AtoB = define_generator(midi_shape)    # generator: A -> B
    g_model_BtoA = define_generator(midi_shape)    # generator: B -> A

    # tie generator weights together
    #tie_weights(g_model_AtoB, g_model_BtoA, 64, 7, n_samples)
    #tie_weights(g_model_AtoB, g_model_BtoA, 67, 4, n_samples)
    #tie_weights(g_model_AtoB, g_model_BtoA, 70, 1, n_samples)

    #tie_weights(g_model_BtoA, g_model_AtoB, 64, 7, n_samples)
    #tie_weights(g_model_BtoA, g_model_AtoB, 67, 4, n_samples)
    #tie_weights(g_model_BtoA, g_model_AtoB, 70, 1, n_samples)

    # define discriminator
    d_model_A = define_discriminator(midi_shape)   # discriminator: A -> [real/fake]
    d_model_B = define_discriminator(midi_shape)   # discriminator: B -> [real/fake]

    # define composite models
    c_model_AtoBtoA = define_composite_model(midi_shape, g_model_AtoB, d_model_B, g_model_BtoA)   # composite: A -> B -> [real/fake, A]
    c_model_BtoAtoB = define_composite_model(midi_shape, g_model_BtoA, d_model_A, g_model_AtoB)   # composite: B -> A -> [real/fake, B]

    # create datasets
    datasetFileNamesA, datasetFileNamesB = load_dataset_filenames()

    # train
    train(d_model_A, d_model_B, g_model_AtoB, g_model_BtoA, c_model_AtoBtoA, c_model_BtoAtoB, (datasetFileNamesA, datasetFileNamesB))

"""
This is a practical usage of the models. If the program is not set to train, this will be called. Loads saved models
and generates MIDI files.
"""
def generate_midis():
    # define models
    g_model_AtoB = define_generator(midi_shape)
    g_model_BtoA = define_generator(midi_shape)

    fA, fB = load_dataset_filenames()
    song, label = generate_real_samples(fA, 1)
    transformed_song, _ = generate_fake_samples(g_model_AtoB, song)
    retransformed, _ = generate_fake_samples(g_model_BtoA, song)

    song.reshape((NUMBER_OF_SAMPLES, NUM_OCTAVES, NOTES_PER_OCTAVE))
    transformed_song = transformed_song[0].reshape((NUMBER_OF_SAMPLES, NUM_OCTAVES, NOTES_PER_OCTAVE))
    retransformed = retransformed[0].reshape((NUMBER_OF_SAMPLES, NUM_OCTAVES, NOTES_PER_OCTAVE))

    representation_to_midi(song.reshape((1, NUMBER_OF_SAMPLES, 8, 12, 1)), "original")
    representation_to_midi(transformed_song.reshape(1, NUMBER_OF_SAMPLES, 8, 12, 1),  "new")

    # compare cross-section of input with both outputs
    plt.imshow(song[0][:])
    plt.savefig("tttt_original")
    plt.clf()

    plt.imshow(transformed_song[0][:])
    plt.savefig("tttt_fake_AB")
    plt.clf()

    plt.imshow(retransformed[0][:])
    plt.savefig("tttt_fake_BA")
    plt.clf()


if __name__ == "__main__":
    if trainModels:
        main()
    else:
        generate_midis()