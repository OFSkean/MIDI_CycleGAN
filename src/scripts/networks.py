from keras.optimizers import Adam
from keras.initializers import RandomNormal
from keras.models import Model
from keras.models import Input
from keras.layers import Conv3D
from keras.layers import Conv3DTranspose
from keras.layers import LeakyReLU
from keras.layers import Activation
from keras.layers import Add
from keras.layers import Layer
import keras
import tensorflow as tf
from keras_contrib.layers.normalization.instancenormalization import InstanceNormalization

# define the discriminator model
def define_discriminator(input_shape):
    # weight initialization
    init = RandomNormal(stddev=0.02)

    # source image input
    in_image = Input(shape=input_shape)

    # C64
    d = Conv3D(64, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=init)(in_image)
    d = LeakyReLU(alpha=0.2)(d)

    # C128
    d = Conv3D(128, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C256
    d = Conv3D(256, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # C512
    d = Conv3D(512, (4, 4, 4), strides=(2, 2, 2), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # second last output layer
    d = Conv3D(512, (4, 4, 4), padding='same', kernel_initializer=init)(d)
    d = InstanceNormalization(axis=-1)(d)
    d = LeakyReLU(alpha=0.2)(d)

    # output
    d = Conv3D(1, (2, 2, 2), strides=(25, 2, 2), padding='same', kernel_initializer=init)(d)

    # define model
    model = Model(in_image, d)

    # compile model
    model.compile(loss='mse', optimizer=Adam(lr=0.0002, beta_1=0.5), loss_weights=[0.5])
    return model


# define a resnet block
def resnet_block(n_filters, input_layer):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # first layer convolutional layer
    g = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer=init)(input_layer)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # second convolutional layer
    g = Conv3D(n_filters, (3, 3, 3), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    # add merge channel-wise with input layer
    g = Add()([g, input_layer])
    return g

# define the standalone generator model
def define_generator(input_shape, n_resnet=9):
    # weight initialization
    init = RandomNormal(stddev=0.02)
    # image input
    in_image = Input(shape=input_shape)
    # c7s1-64
    g = Conv3D(64, (7, 7, 7), padding='same', kernel_initializer=init)(in_image)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d128
    g = Conv3D(128, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)
    # d256
    g = Conv3D(256, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # R256
    for _ in range(n_resnet):
        g = resnet_block(256, g)

    # u128
    g = Conv3DTranspose(128, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # u64
    g = Conv3DTranspose(64, (3, 3, 3), strides=(2, 2, 2), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    g = Activation('relu')(g)

    # c7s1-3
    g = Conv3D(1, (7, 7, 7), padding='same', kernel_initializer=init)(g)
    g = InstanceNormalization(axis=-1)(g)
    out_image = Activation('relu')(g)

    # define model
    model = Model(in_image, out_image)
    return model

# define the GAN
def define_composite_model(midi_shape, g_model_1, d_model, g_model_2):
    # ensure the model we're updating is trainable
    g_model_1.trainable = True
    # mark discriminator as not trainable
    d_model.trainable = False
    # mark other generator model as not trainable
    g_model_2.trainable = False

    # discriminator element
    input_gen = Input(shape=midi_shape)

    gen1_out = g_model_1(input_gen)

    output_d = d_model(gen1_out)

    # identity element
    input_id = Input(shape=midi_shape)
    output_id = g_model_1(input_id)
    # forward cycle
    output_f = g_model_2(gen1_out)
    # backward cycle
    gen2_out = g_model_2(input_id)
    output_b = g_model_1(gen2_out)
    # define model graph
    model = Model([input_gen, input_id], [output_d, output_id, output_f, output_b])
    # define optimization algorithm configuration
    opt = Adam(lr=0.0002, beta_1=0.5)
    # compile model with weighting of least squares loss and L1 loss
    model.compile(loss=['mse', 'mae', 'mae', 'mae'], loss_weights=[1, 5, 10, 10], optimizer=opt)
    return model