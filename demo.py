"""
A quick demo of paper 'illumination coding meets uncertainty learning: toward reliable AI-augmented
 phase imaging'.  (https://arxiv.org/abs/1901.02038)
 Please consider citing our paper if you find the script useful in your own research projects.

Pretrained model is trained on Hela (fixed in ethanol) data along and performs predictions on both seen and unseen
cell types (Hela fixed in ethanol and formalin)

-Yujia Xue
Computational Imaging Systems Lab (http://sites.bu.edu/tianlab/)
April 2019
Boston University, ECE department
"""

import numpy as np
import matplotlib.pyplot as plt
from model import get_model_dropout_activated

num_dropout_ensembles = 16
num_examples = 3
patch_dim = 384
model = get_model_dropout_activated(input_shape=(384, 384, 5), l2_weight_decay=1e-6, DO_rate=0.5)
model.load_weights('pretrained_model_hela_ethanol.hdf5')

data_hela_ethanol = np.load('test_data/ethanol.npy')
data_hela_formalin = np.load('test_data/formalin.npy')

measurement_hela_ethanol = data_hela_ethanol[:, :, :, :5]
measurement_hela_formalin = data_hela_formalin[:, :, :, :5]
gt_hela_ethanol = data_hela_ethanol[:, :, :, 5]
gt_hela_formalin = data_hela_formalin[:, :, :, 5]

ethanol_prediction_ensembles = np.ndarray((num_examples, patch_dim, patch_dim, 2, num_dropout_ensembles))
formalin_prediction_ensembles = np.ndarray((num_examples, patch_dim, patch_dim, 2, num_dropout_ensembles))
for dropout_idx in range(num_dropout_ensembles):
    tmp = model.predict(measurement_hela_ethanol, batch_size=2)
    ethanol_prediction_ensembles[:, :, :, :, dropout_idx] = tmp
    tmp = model.predict(measurement_hela_formalin, batch_size=2)
    formalin_prediction_ensembles[:, :, :, :, dropout_idx] = tmp
    print('dropout ensembles: ' + str(dropout_idx + 1) + '/' + str(num_dropout_ensembles))

ethanol_result = np.ndarray((patch_dim, patch_dim, 2, num_examples))
formalin_result = np.ndarray((patch_dim, patch_dim, 2, num_examples))
for i in range(num_examples):
    ethanol_result[:, :, 0, i] = np.mean(ethanol_prediction_ensembles[i, :, :, 0, :].squeeze(), axis=2)
    formalin_result[:, :, 0, i] = np.mean(formalin_prediction_ensembles[i, :, :, 0, :].squeeze(), axis=2)
    ethanol_result[:, :, 1, i] = np.sqrt((np.mean(ethanol_prediction_ensembles[i, :, :, 1, :].squeeze(), axis=2)) ** 2 +
                                         (np.std(ethanol_prediction_ensembles[i, :, :, 0, :].squeeze(), axis=2)) ** 2)
    formalin_result[:, :, 1, i] = np.sqrt(
        (np.mean(formalin_prediction_ensembles[i, :, :, 1, :].squeeze(), axis=2)) ** 2 +
        (np.std(formalin_prediction_ensembles[i, :, :, 0, :].squeeze(), axis=2)) ** 2)

plt.figure(figsize=[20, 8])
for i in range(3):
    handle = plt.subplot(2, 6, 1 + 2 * i)
    handle.set_title('predicted phase (seen cell type)')
    plt.imshow(ethanol_result[:, :, 0, i], vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    handle = plt.subplot(2, 6, 2 + 2 * i)
    handle.set_title('predicted uncertainty')
    plt.imshow(ethanol_result[:, :, 1, i], cmap='jet', vmin=0, vmax=0.15)
    plt.axis('off')
    handle = plt.subplot(2, 6, 7 + 2 * i)
    handle.set_title('ground truth')
    plt.imshow(gt_hela_ethanol[i, :, :].squeeze(), vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    handle = plt.subplot(2, 6, 8 + 2 * i)
    handle.set_title('absolute error')
    plt.imshow(np.abs(gt_hela_ethanol[i, :, :].squeeze() - ethanol_result[:, :, 0, i]), cmap='jet', vmin=0, vmax=0.45)
    plt.axis('off')

plt.figure(figsize=[20, 8])
for i in range(3):
    handle = plt.subplot(2, 6, 1 + 2 * i)
    handle.set_title('predicted phase (unseen cell type)')
    plt.imshow(formalin_result[:, :, 0, i], vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    handle = plt.subplot(2, 6, 2 + 2 * i)
    handle.set_title('predicted uncertainty')
    plt.imshow(formalin_result[:, :, 1, i], cmap='jet', vmin=0, vmax=0.15)
    plt.axis('off')
    handle = plt.subplot(2, 6, 7 + 2 * i)
    handle.set_title('ground truth')
    plt.imshow(gt_hela_formalin[i, :, :].squeeze(), vmin=0, vmax=1, cmap='gray')
    plt.axis('off')
    handle = plt.subplot(2, 6, 8 + 2 * i)
    handle.set_title('absolute error')
    plt.imshow(np.abs(gt_hela_formalin[i, :, :].squeeze() - formalin_result[:, :, 0, i]), cmap='jet', vmin=0, vmax=0.45)
    plt.axis('off')

plt.show()
