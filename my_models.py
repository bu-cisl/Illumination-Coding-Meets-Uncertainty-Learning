from __future__ import print_function

import keras
from keras.layers import AveragePooling2D, Lambda
import keras.backend as K
from keras.layers import Input, MaxPooling2D, UpSampling2D, Dropout, Conv2D, Concatenate, Activation, Cropping2D, \
    Flatten, Dense, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import add
from keras.layers.normalization import BatchNormalization
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.regularizers import l2, l1
from keras.activations import sigmoid, relu

img_rows = 512
img_cols = 512
save_path = 'save/'
num_epochs = 1
save_period = 10
show_groundtruth_flag = False


def _bn_relu(input):
    norm = BatchNormalization(axis=-1)(input)
    return Activation("relu")(norm)


def _bn_relu_conv(**conv_params):
    """Helper to build a BN -> relu -> conv block.
    This is an improved scheme proposed in http://arxiv.org/pdf/1603.05027v2.pdf
    """
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        activation = _bn_relu(input)
        return Conv2D(filters=filters, kernel_size=kernel_size,
                      strides=strides, padding=padding,
                      kernel_initializer=kernel_initializer,
                      kernel_regularizer=kernel_regularizer)(activation)

    return f


def _double_bn_relu_conv(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        after_first_bn_relu_conv = _bn_relu_conv(filters=filters, kernel_size=kernel_size,
                                                 strides=strides, kernel_initializer=kernel_initializer,
                                                 padding=padding, kernel_regularizer=kernel_regularizer)(input)
        return _bn_relu_conv(filters=filters, kernel_size=kernel_size,
                             strides=strides, kernel_initializer=kernel_initializer,
                             padding=padding, kernel_regularizer=kernel_regularizer)(after_first_bn_relu_conv)

    return f


def res_block(**conv_params):
    filters = conv_params["filters"]
    kernel_size = conv_params["kernel_size"]
    strides = conv_params.setdefault("strides", (1, 1))
    kernel_initializer = conv_params.setdefault("kernel_initializer", "he_normal")
    padding = conv_params.setdefault("padding", "same")
    kernel_regularizer = conv_params.setdefault("kernel_regularizer", l2(1.e-4))

    def f(input):
        after_double_bn_relu_conv = _double_bn_relu_conv(filters=filters, kernel_size=kernel_size,
                                                         strides=strides, kernel_initializer=kernel_initializer,
                                                         padding=padding, kernel_regularizer=kernel_regularizer)(input)
        return add([input, after_double_bn_relu_conv])

    return f


def conv_factory(x, concat_axis, nb_filter,
                 dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
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
        x = Dropout(dropout_rate)(x)

    return x


def conv_factory_DO(x, concat_axis, nb_filter,
                    dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
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


def conv_factory_leaky(x, concat_axis, nb_filter,
                       dropout_rate=None, weight_decay=1E-4):
    """Apply BatchNorm, Relu 3x3Conv2D, optional dropout
    :param x: Input keras network
    :param concat_axis: int -- index of contatenate axis
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras network with b_norm, relu and Conv2D added
    :rtype: keras network
    """

    x = BatchNormalization(axis=concat_axis,
                           gamma_regularizer=l2(weight_decay),
                           beta_regularizer=l2(weight_decay))(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(nb_filter, (3, 3),
               kernel_initializer="he_uniform",
               padding="same",
               kernel_regularizer=l2(weight_decay))(x)
    if dropout_rate:
        x = Dropout(dropout_rate)(x)

    return x


def denseblock(x, concat_axis, nb_layers, growth_rate,
               dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory(x, concat_axis, growth_rate,
                         dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x


def denseblock_DO(x, concat_axis, nb_layers, growth_rate,
                  dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory_DO(x, concat_axis, growth_rate,
                            dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x


def denseblock_leaky(x, concat_axis, nb_layers, growth_rate,
                     dropout_rate=None, weight_decay=1E-4):
    """Build a denseblock where the output of each
       conv_factory is fed to subsequent ones
    :param x: keras model
    :param concat_axis: int -- index of contatenate axis
    :param nb_layers: int -- the number of layers of conv_
                      factory to append to the model.
    :param nb_filter: int -- number of filters
    :param dropout_rate: int -- dropout rate
    :param weight_decay: int -- weight decay factor
    :returns: keras model with nb_layers of conv_factory appended
    :rtype: keras model
    """
    list_feat = [x]
    for i in range(nb_layers):
        x = conv_factory_leaky(x, concat_axis, growth_rate,
                               dropout_rate, weight_decay)
        list_feat.append(x)
        x = Concatenate(axis=concat_axis)(list_feat)

    return x


def discriminator_96(input_shape):
    img_shape = input_shape
    model = Sequential()
    model.add(Conv2D(64, kernel_size=5, strides=2, input_shape=img_shape, padding='valid',
                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Conv2D(64, kernel_size=5, strides=2, padding="valid",
                     kernel_initializer=keras.initializers.RandomNormal(mean=0.0, stddev=0.02, seed=None)))
    model.add(BatchNormalization(momentum=0.99))
    model.add(LeakyReLU(alpha=0.2))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.4))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(rate=0.4))
    model.add(Dense(1, activation='sigmoid'))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)


def get_model_sigmoid_2out(input_shape, l2_weight_decay):
    regularizer_func = l2(l2_weight_decay)

    inputs = Input(input_shape)
    print("inputs shape:", inputs.shape)
    conv1 = Conv2D(64, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(inputs)
    print("conv1 shape:", conv1.shape)
    # conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    # print("conv1 shape:", conv1.shape)
    # res1 = res_block(filters=64, kernel_size=3)(conv1)
    # print("res1 shape:", res1.shape)
    db1 = denseblock(x=conv1, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db1 shape:", db1.shape)
    pool1 = MaxPooling2D(pool_size=(2, 2))(db1)
    print("pool1 shape:", pool1.shape)

    conv2 = Conv2D(128, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(pool1)
    print("conv2 shape:", conv2.shape)
    # conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    # print("conv2 shape:", conv2.shape)
    # res2 = res_block(filters=128, kernel_size=3)(conv2)
    # print("res2 shape:", res2.shape)

    db2 = denseblock(x=conv2, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db2 shape:", db2.shape)
    pool2 = MaxPooling2D(pool_size=(2, 2))(db2)
    print("pool2 shape:", pool2.shape)

    conv3 = Conv2D(256, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(pool2)
    print("conv3 shape:", conv3.shape)
    # conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    # print("conv3 shape:", conv3.shape)
    # res3 = res_block(filters=256,kernel_size=3)(conv3)
    # print("res3 shape:", res3.shape)

    db3 = denseblock(x=conv3, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db3 shape:", db3.shape)
    pool3 = MaxPooling2D(pool_size=(2, 2))(db3)
    print("pool3 shape:", pool3.shape)

    conv4 = Conv2D(512, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(pool3)
    print("conv4 shape:", conv4.shape)
    # conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    # print("conv4 shape:", conv4.shape)
    # res4 = res_block(filters=512, kernel_size=3)(conv4)
    # print("res4 shape:", res4.shape)

    db4 = denseblock(x=conv4, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db4 shape:", db4.shape)
    drop4 = Dropout(0.5)(db4)
    print("drop4 shape:", drop4.shape)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    print("pool4 shape:", pool4.shape)

    conv5 = Conv2D(1024, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(pool4)
    print("conv5 shape:", conv5.shape)
    # conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    # print("conv5 shape:", conv5.shape)
    # res5 = res_block(filters=1024,kernel_size=3)(conv5)
    # print("res5 shape:", res5.shape)
    db5 = denseblock(x=conv5, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db5 shape:", db5.shape)
    drop5 = Dropout(0.5)(db5)
    print("drop5 shape:", drop5.shape)

    up6 = Conv2D(512, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularizer_func)(
        UpSampling2D(size=(2, 2))(drop5))
    print("up6 shape:", up6.shape)
    merge6 = Concatenate(axis=3)([drop4, up6])
    print("merge6 shape:", merge6.shape)
    conv6 = Conv2D(512, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(merge6)
    print("conv6 shape:", conv6.shape)
    # conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
    # print("conv6 shape:", conv6.shape)
    # res5 = res_block(filters=512, kernel_size=3)(conv6)
    # print("res5 shape:", res5.shape)
    db6 = denseblock(x=conv6, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db6 shape:", db6.shape)

    up7 = Conv2D(256, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularizer_func)(
        UpSampling2D(size=(2, 2))(db6))
    print("up7 shape:", up7.shape)
    merge7 = Concatenate(axis=3)([db3, up7])
    print("merge7 shape:", merge7.shape)
    conv7 = Conv2D(256, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(merge7)
    print("conv7 shape:", conv7.shape)
    # conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
    # print("conv7 shape:", conv7.shape)
    # res6 = res_block(filters=256, kernel_size=3)(conv7)
    # print("res6 shape:", res6.shape)
    db7 = denseblock(x=conv7, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db7 shape:", db7.shape)

    up8 = Conv2D(128, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularizer_func)(
        UpSampling2D(size=(2, 2))(db7))
    print("up8 shape:", up8.shape)
    merge8 = Concatenate(axis=3)([db2, up8])
    print("merge8 shape:", merge8.shape)
    conv8 = Conv2D(128, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(merge8)
    print("conv8 shape:", conv8.shape)
    # conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
    # print("conv8 shape:", conv8.shape)
    # res7 = res_block(filters=128, kernel_size=3)(conv8)
    # print("res7 shape:", res7.shape)
    db8 = denseblock(x=conv8, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db8 shape:", db8.shape)

    up9 = Conv2D(64, 2, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                 kernel_regularizer=regularizer_func)(
        UpSampling2D(size=(2, 2))(db8))
    print("up9 shape:", up9.shape)
    merge9 = Concatenate(axis=3)([db1, up9])  ##res1 up9
    print("merge9 shape:", merge9.shape)
    conv9 = Conv2D(64, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                   kernel_regularizer=regularizer_func)(merge9)
    print("conv9 shape:", conv9.shape)
    # conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    # print("conv9 shape:", conv9.shape)
    # res8 = res_block(filters=64, kernel_size=3)(conv9)
    # print("res8 shape:", res8.shape)
    db9 = denseblock(x=conv9, concat_axis=3, nb_layers=3, growth_rate=16, dropout_rate=0.5,
                     weight_decay=l2_weight_decay)
    print("db9 shape:", db9.shape)
    conv10 = Conv2D(16, 3, activation=LeakyReLU(0.2), padding='same', kernel_initializer='he_normal',
                    kernel_regularizer=regularizer_func)(db9)
    print("conv10 shape:", conv9.shape)
    conv11 = Conv2D(2, 1, activation='sigmoid', kernel_regularizer=regularizer_func)(conv10)
    print("conv11 shape:", conv11.shape)

    model = Model(inputs=inputs, outputs=conv11)
    return model
