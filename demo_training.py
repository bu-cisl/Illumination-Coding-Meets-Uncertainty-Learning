"""
A demo training code of paper 'Reliable deep learning-based phase imaging with uncertainty quantification'.  (https://www.osapublishing.org/optica/abstract.cfm?uri=optica-6-5-618)
 Please consider citing our paper if you find the script useful in your own research projects.

-Yujia Xue
Computational Imaging Systems Lab (http://sites.bu.edu/tianlab/)
July 2021
Boston University, ECE department
"""

from __future__ import print_function

import keras.backend as K
import matplotlib.pyplot as plt
import numpy as np
from keras import Model
from keras.layers import Input, Lambda, Average
from keras.optimizers import Adam

from my_losses import my_accuracy, laplacian_loss, mae_on_first_channel
from my_models import get_model_sigmoid_2out, discriminator_96

plt.switch_backend('agg')


def dice_image(input_image_stack, nb_dices, dim_dices):
    output_stack = input_image_stack[:, 0 * dim_dices:1 * dim_dices, 0 * dim_dices:1 * dim_dices, :]
    for i in range(nb_dices):
        for j in range(nb_dices):
            if i + j != 0:
                output_stack = np.append(output_stack, input_image_stack[:, i * dim_dices:i * dim_dices + dim_dices,
                                                       j * dim_dices:j * dim_dices + dim_dices, :], axis=0)
    return output_stack


def real_labels_1channel(length, val_real):
    return val_real * np.ones((length, 1))


def fake_labels_1channel(length, val_fake):
    return val_fake * np.ones((length, 1))


def shuffle_renew_fake_imgs(old_stack, new_stack):
    half_size = int(np.round(old_stack.shape[0] / 2))
    indices = np.random.choice(old_stack.shape[0] - 1, half_size, replace=False)
    output_stack = old_stack[indices, :]
    indices = np.random.choice(old_stack.shape[0] - 1, old_stack.shape[0] - half_size, replace=False)
    output_stack = np.concatenate((output_stack, new_stack[indices, :]), axis=0)
    return output_stack


img_rows = 384
img_cols = 384
save_path = 'save/'
save_img_path = 'save/imgs/'

# tbd
batch_size_gan = 4  # local GPU:2 SCC:4
batch_size_d = 128
terminate_d_threshold = 0.7
proj_name = 'formalin_hela'
save_model_period = 10
save_image_period = 2

prototyping_flag = False
if prototyping_flag:
    num_iters = 3
else:
    num_iters = 400  # 500

# load data files, change to your own data, arranged as batch * rows * cols * channels
x_train = np.load('processed_data/x_train_formalin.npy')
y_train = np.load('processed_data/y_train_formalin.npy')
x_test = np.load('processed_data/x_test_formalin.npy')
y_test = np.load('processed_data/y_test_formalin.npy')

num_train_sample = x_train.shape[0]
num_test_sample = x_test.shape[0]


def lr_schedule(current_iter, initial_lr, decay_factor, decay_period):
    num_decays = np.floor(current_iter / decay_period)
    updated_lr = initial_lr * np.power(decay_factor, num_decays)
    return updated_lr


# training settings
D_inital_lr = 2e-5
D_decay_factor = 1
D_decay_period = 1000000
G_inital_lr = 2e-5
G_decay_factor = 0.5
G_decay_period = 100

# setup GAN
D = discriminator_96(input_shape=(96, 96, 1))
opt_D = Adam(lr=lr_schedule(0, D_inital_lr, D_decay_factor, D_decay_period), beta_1=0.5, beta_2=0.999, epsilon=1e-08)
D.compile(loss='binary_crossentropy', optimizer=opt_D, metrics=[my_accuracy])
D.summary()

G_input = Input(shape=[img_rows, img_cols, 5], name="GAN_input")
G = get_model_sigmoid_2out(input_shape=(img_rows, img_cols, 5), l2_weight_decay=1e-6)
generated_image = G(G_input)
generated_image_mean = Lambda(lambda x: K.expand_dims(x[:, :, :, 0], axis=-1))(generated_image)
scaled_generated_img = Lambda(lambda x: 2.0 * x - 1.0)(generated_image_mean)
D.trainable = False

# dice into small patches
slice_11 = Lambda(lambda x: x[:, 96 * 0:96 * 1, 96 * 0:96 * 1, :])(scaled_generated_img)
slice_12 = Lambda(lambda x: x[:, 96 * 0:96 * 1, 96 * 1:96 * 2, :])(scaled_generated_img)
slice_13 = Lambda(lambda x: x[:, 96 * 0:96 * 1, 96 * 2:96 * 3, :])(scaled_generated_img)
slice_14 = Lambda(lambda x: x[:, 96 * 0:96 * 1, 96 * 3:96 * 4, :])(scaled_generated_img)

slice_21 = Lambda(lambda x: x[:, 96 * 1:96 * 2, 96 * 0:96 * 1, :])(scaled_generated_img)
slice_22 = Lambda(lambda x: x[:, 96 * 1:96 * 2, 96 * 1:96 * 2, :])(scaled_generated_img)
slice_23 = Lambda(lambda x: x[:, 96 * 1:96 * 2, 96 * 2:96 * 3, :])(scaled_generated_img)
slice_24 = Lambda(lambda x: x[:, 96 * 1:96 * 2, 96 * 3:96 * 4, :])(scaled_generated_img)

slice_31 = Lambda(lambda x: x[:, 96 * 2:96 * 3, 96 * 0:96 * 1, :])(scaled_generated_img)
slice_32 = Lambda(lambda x: x[:, 96 * 2:96 * 3, 96 * 1:96 * 2, :])(scaled_generated_img)
slice_33 = Lambda(lambda x: x[:, 96 * 2:96 * 3, 96 * 2:96 * 3, :])(scaled_generated_img)
slice_34 = Lambda(lambda x: x[:, 96 * 2:96 * 3, 96 * 3:96 * 4, :])(scaled_generated_img)

slice_41 = Lambda(lambda x: x[:, 96 * 3:96 * 4, 96 * 0:96 * 1, :])(scaled_generated_img)
slice_42 = Lambda(lambda x: x[:, 96 * 3:96 * 4, 96 * 1:96 * 2, :])(scaled_generated_img)
slice_43 = Lambda(lambda x: x[:, 96 * 3:96 * 4, 96 * 2:96 * 3, :])(scaled_generated_img)
slice_44 = Lambda(lambda x: x[:, 96 * 3:96 * 4, 96 * 3:96 * 4, :])(scaled_generated_img)

D_out_11 = D(slice_11)
D_out_12 = D(slice_12)
D_out_13 = D(slice_13)
D_out_14 = D(slice_14)

D_out_21 = D(slice_21)
D_out_22 = D(slice_22)
D_out_23 = D(slice_23)
D_out_24 = D(slice_24)

D_out_31 = D(slice_31)
D_out_32 = D(slice_32)
D_out_33 = D(slice_33)
D_out_34 = D(slice_34)

D_out_41 = D(slice_41)
D_out_42 = D(slice_42)
D_out_43 = D(slice_43)
D_out_44 = D(slice_44)

GAN_output = Average()([D_out_11, D_out_12, D_out_13, D_out_14,
                        D_out_21, D_out_22, D_out_23, D_out_24,
                        D_out_31, D_out_32, D_out_33, D_out_34,
                        D_out_41, D_out_42, D_out_43, D_out_44])

GAN = Model(inputs=[G_input],
            outputs=[generated_image, GAN_output],
            name="GAN")
GAN_loss = [laplacian_loss, 'binary_crossentropy']
opt_GAN = Adam(lr=lr_schedule(0, G_inital_lr, G_decay_factor, G_decay_period), beta_1=0.9, beta_2=0.999, epsilon=1e-08)
loss_weights = [1, 0.005]
GAN.compile(loss=GAN_loss, loss_weights=loss_weights, optimizer=opt_GAN, metrics={'model_1': mae_on_first_channel})
GAN.summary()

# training start here
real_val = 1.0
fake_val = 0.0

# can load pretrained models here, not necessary
# D.load_weights('save/pre_D.hdf5')
G.load_weights('save/formalin_g_G_500.hdf5')

for iteration in range(0, num_iters, 1):
    # train D until D can distinguish real and generated images
    lr_D = lr_schedule(iteration, D_inital_lr, D_decay_factor, D_decay_period)
    K.set_value(D.optimizer.lr, lr_D)

    if iteration == 0:
        gen_imgs = np.expand_dims(G.predict(x_train, batch_size=batch_size_gan)[:, :, :, 0], axis=-1)
    else:
        gen_imgs = shuffle_renew_fake_imgs(gen_imgs,
                                           np.expand_dims(G.predict(x_train, batch_size=batch_size_gan)[:, :, :, 0],
                                                          axis=-1))

    real_imgs = dice_image(y_train * 2.0 - 1.0, 4, 96)
    fake_imgs = dice_image(gen_imgs * 2.0 - 1.0, 4, 96)
    D_input = np.concatenate((real_imgs, fake_imgs), axis=0)
    D_output = np.concatenate((real_labels_1channel(real_imgs.shape[0], real_val),
                               fake_labels_1channel(fake_imgs.shape[0], fake_val)), axis=0)

    if iteration == 0:
        gen_imgs_val = np.expand_dims(G.predict(x_test, batch_size=batch_size_gan)[:, :, :, 0], axis=-1)
    else:
        gen_imgs_val = shuffle_renew_fake_imgs(gen_imgs_val,
                                               np.expand_dims(G.predict(x_test, batch_size=batch_size_gan)[:, :, :, 0],
                                                              axis=-1))
    real_imgs_val = dice_image(y_test * 2.0 - 1.0, 4, 96)
    fake_imgs_val = dice_image(gen_imgs_val * 2.0 - 1.0, 4, 96)
    D_input_val = np.concatenate((real_imgs_val, fake_imgs_val), axis=0)
    D_output_val = np.concatenate((real_labels_1channel(real_imgs_val.shape[0], real_val),
                                   fake_labels_1channel(fake_imgs_val.shape[0], fake_val)), axis=0)

    flag_train_D = True
    epochs_count = 0
    while flag_train_D:
        D.fit(x=D_input, y=D_output, epochs=1, batch_size=batch_size_d, shuffle=True, verbose=2)
        epochs_count = epochs_count + 1
        loss_d_val = D.evaluate(x=D_input_val, y=D_output_val, verbose=0)
        if loss_d_val[1] >= terminate_d_threshold:
            flag_train_D = False
            txt_file = open('log_' + proj_name + '.txt', "a")
            txt_file.write('Iter: ' + str(iteration + 1) + ', Epochs: ' + str(epochs_count) + ', acc on val: ' +
                           str(np.round(loss_d_val[1], 4)) + ', terminate D sub-iter training. \n')
            txt_file.close()
        else:
            txt_file = open('log_' + proj_name + '.txt', "a")
            txt_file.write('Iter: ' + str(iteration + 1) + ', Epochs: ' + str(epochs_count) + ', acc on val: ' +
                           str(np.round(loss_d_val[1], 4)) + ', continue D sub-iter training. \n')
            txt_file.close()

    # train G until GAN outputs average scores is above threshold
    lr_G = lr_schedule(iteration, G_inital_lr, G_decay_factor, G_decay_period)
    K.set_value(GAN.optimizer.lr, lr_G)

    GAN.fit(x=x_train, y=[y_train, real_labels_1channel(y_train.shape[0], real_val)], epochs=1,
            batch_size=batch_size_gan, shuffle=True, verbose=2)
    [pred_val, labels_val] = GAN.predict(x=x_test, batch_size=batch_size_gan)
    mae_val = np.mean(np.abs(pred_val[:, :, :, 0] - y_test.squeeze()))
    average_score_val = np.mean(labels_val)
    txt_file = open('log_' + proj_name + '.txt', "a")
    txt_file.write('Iter: ' + str(iteration + 1) + ', mae on val: ' +
                   str(np.round(mae_val, 4)) + ', score on val: ' +
                   str(np.round(average_score_val, 4)) + '\n')
    txt_file.close()

    if (iteration + 1) % save_model_period == 0:
        G.save_weights(save_path + proj_name + '_G_' + str(iteration + 1) + '.hdf5')
        D.save_weights(save_path + proj_name + '_D_' + str(iteration + 1) + '.hdf5')

    if (iteration + 1) % save_image_period == 0:
        for i in range(5):
            plt.figure(figsize=[12, 8])

            handle = plt.subplot(2, 3, 1)
            plt.imshow(y_test[10 * i, :].squeeze(), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            handle.set_title('GT (cmap: gray)')

            handle = plt.subplot(2, 3, 4)
            plt.imshow(y_test[10 * i, :].squeeze(), vmin=0, vmax=1, cmap='viridis')
            plt.axis('off')
            handle.set_title('GT (cmap: Viridis)')

            handle = plt.subplot(2, 3, 2)
            plt.imshow(pred_val[10 * i, :, :, 0].squeeze(), vmin=0, vmax=1, cmap='gray')
            plt.axis('off')
            handle.set_title('Predict (cmap: gray)')

            handle = plt.subplot(2, 3, 5)
            plt.imshow(pred_val[10 * i, :, :, 0].squeeze(), vmin=0, vmax=1, cmap='viridis')
            plt.axis('off')
            handle.set_title('Predict (cmap: Viridis)')

            handle = plt.subplot(2, 3, 3)
            plt.imshow(pred_val[10 * i, :, :, 1].squeeze(), cmap='viridis')
            plt.axis('off')
            handle.set_title('Uncertainty')

            handle = plt.subplot(2, 3, 6)
            plt.imshow(np.abs(pred_val[10 * i, :, :, 0].squeeze() - y_test[10 * i, :].squeeze()), cmap='viridis')
            plt.axis('off')
            handle.set_title('Absolute error')

            plt.tight_layout()
            plt.savefig(save_img_path + proj_name + '_' + str(iteration + 1) + '_' + str(i + 1) + '.png')
            plt.close()
