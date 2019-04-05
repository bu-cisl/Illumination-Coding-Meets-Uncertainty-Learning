"""
Keras implementation of model in paper 'illumination coding meets uncertainty learning: toward reliable AI-augmented
 phase imaging'.  (https://arxiv.org/abs/1901.02038)
 Please consider citing our paper if you find the script useful in your own research projects.

-Yujia Xue
Computational Imaging Systems Lab (http://sites.bu.edu/tianlab/)
April 2019
Boston University, ECE department
"""

from __future__ import print_function
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.regularizers import l2


def conv_factory_dropout(x, concat_axis, nb_filter,
                         dropout_rate=None, weight_decay=1E-4):
    """
    convolution factory with dropout activated in prediction process
    :param x: input layer
    :param concat_axis: along which axis to perfrom batch normalization
    :param nb_filter: number of filters
    :param dropout_rate: rate of dropout
    :param weight_decay: l2 weight regularization parameter
    :return: a keras layer
    """
    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = Activation('relu')(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x, training=True)

    return x


def denseblock_dropout(x, concat_axis, nb_layers, growth_rate,
                       dropout_rate=None, weight_decay=1E-4):
    """
    denseblock with dropout activated in prediction process
    :param x: input layer
    :param concat_axis: along which axis to concatenate layers
    :param nb_layers: number of layers within the denseblock
    :param growth_rate: number of filters in each convolution factory (also known as growth rate of denseblock)
    :param dropout_rate: rate of dropout
    :param weight_decay:  l2 weight regularization parameter
    :return: a keras layer
    """
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory_dropout(x, concat_axis, growth_rate,
                                 dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x


def get_model_dropout_activated(input_shape, l2_weight_decay, DO_rate):
    """
    generate a network with dropout layers activated in the prediction process
    :param input_shape: shape of input layer
    :param l2_weight_decay: l2 weight regularization parameter
    :param DO_rate: dropout rate
    :return: a keras model
    """
    regularization_function = l2(l2_weight_decay)

    inputs = Input(input_shape)
    print("inputs shape:", inputs.shape)
    conv1 = Conv2D(64, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(inputs)
    print("conv1 shape:", conv1.shape)
    db1 = denseblock_dropout(x=conv1, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)
    conv2 = Conv2D(128, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(pool1)
    print("conv2 shape:", conv2.shape)
    db2 = denseblock_dropout(x=conv2, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)
    conv3 = Conv2D(256, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(pool2)
    print("conv3 shape:", conv3.shape)
    db3 = denseblock_dropout(x=conv3, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db3 shape:", db3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
    print("pool3 shape:", pool3.shape)
    conv4 = Conv2D(512, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(pool3)
    print("conv4 shape:", conv4.shape)
    db4 = denseblock_dropout(x=conv4, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db4 shape:", db4.shape)
    drop4 = Dropout(DO_rate)(db4)
    print("drop4 shape:", drop4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print("pool4 shape:", pool4.shape)
    conv5 = Conv2D(1024, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(pool4)
    print("conv5 shape:", conv5.shape)
    db5 = denseblock_dropout(x=conv5, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db5 shape:", db5.shape)
    drop5 = Dropout(DO_rate)(db5)
    print("drop5 shape:", drop5.shape)
    up6 = Conv2D(512, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularization_function)(
        UpSampling2D(size=(2, 2))(drop5))
    print("up6 shape:", up6.shape)
    merge6 = Concatenate(axis=3)([drop4, up6])
    print("merge6 shape:", merge6.shape)
    conv6 = Conv2D(512, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(merge6)
    print("conv6 shape:", conv6.shape)
    db6 = denseblock_dropout(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db6 shape:", db6.shape)
    up7 = Conv2D(256, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularization_function)(
        UpSampling2D(size=(2, 2))(db6))
    print("up7 shape:", up7.shape)
    merge7 = Concatenate(axis=3)([db3, up7])
    print("merge7 shape:", merge7.shape)
    conv7 = Conv2D(256, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(merge7)
    print("conv7 shape:", conv7.shape)

    db7 = denseblock_dropout(x=conv7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db7 shape:", db7.shape)

    up8 = Conv2D(128, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularization_function)(
        UpSampling2D(size=(2, 2))(db7))
    print("up8 shape:", up8.shape)
    merge8 = Concatenate(axis=3)([db2, up8])
    print("merge8 shape:", merge8.shape)
    conv8 = Conv2D(128, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(merge8)
    print("conv8 shape:", conv8.shape)

    db8 = denseblock_dropout(x=conv8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db8 shape:", db8.shape)

    up9 = Conv2D(64, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularization_function)(
        UpSampling2D(size=(2, 2))(db8))
    print("up9 shape:", up9.shape)
    merge9 = Concatenate(axis=3)([db1, up9])
    print("merge9 shape:", merge9.shape)
    conv9 = Conv2D(64, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularization_function)(merge9)
    print("conv9 shape:", conv9.shape)

    db9 = denseblock_dropout(x=conv9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=DO_rate,
                             weight_decay=l2_weight_decay)
    print("db9 shape:", db9.shape)
    conv10 = Conv2D(16, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularization_function)(db9)
    print("conv10 shape:", conv9.shape)
    conv11 = Conv2D(2, 1, activation='sigmoid', kernel_regularizer=regularization_function)(conv10)
    print("conv11 shape:", conv11.shape)

    model = Model(inputs=inputs, outputs=conv11)
    return model
