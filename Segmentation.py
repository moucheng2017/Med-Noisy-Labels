import os
import errno
import torch
from adamW import AdamW
import timeit
import imageio
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

from Utilis import segmentation_scores, generalized_energy_distance
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from Utilis import CustomDataset_punet, calculate_cm
from Loss import noisy_label_loss_low_rank, noisy_label_loss
from Models import UNet_CMs

from Utilis import evaluate_noisy_label_4, evaluate_noisy_label_5, evaluate_noisy_label_6


def segmentation(model_name, model_path, testdata, class_no, data_set):
    """ This is to generate segmentation maps.

    Args:
        model_name (str): your saved model name
        model_path (str): path to where your model is stored
        testdata (:object, data-loader): testing data loader
        class_no (str):
        data_set (str): dataset tag to specificy which data set, because brats is multi-class and the others are binary,
        so the generated segmentation maps are with different colours

    Returns:

    """
    save_path = '../Exp_Results_Noisy_labels'
    #
    try:
        #
        os.mkdir(save_path)
        #
    except OSError as exc:
        #
        if exc.errno != errno.EEXIST:
            #
            raise
        #
        pass
    #
    save_path = '../Exp_Results_Noisy_labels/' + data_set
    #
    try:
        #
        os.mkdir(save_path)
        #
    except OSError as exc:
        #
        if exc.errno != errno.EEXIST:
            #
            raise
        #
        pass
    #
    save_path = save_path + '/Exp_' + \
                '_Noisy_Label_Net_' + model_name
    #
    try:
        #
        os.mkdir(save_path)
        #
    except OSError as exc:
        #
        if exc.errno != errno.EEXIST:
            #
            raise
        #
        pass
    #
    # for segmentation results generation of our models
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.load(model_path)
    model.eval()
    #
    for i, (v_images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(testdata):
        #
        cm_all_true = []
        #
        cm_over_true = calculate_cm(pred=labels_over, true=labels_good)
        cm_under_true = calculate_cm(pred=labels_under, true=labels_good)
        cm_wrong_true = calculate_cm(pred=labels_wrong, true=labels_good)
        #
        cm_all_true.append(cm_over_true)
        cm_all_true.append(cm_under_true)
        cm_all_true.append(cm_wrong_true)
        #
        # cm_all_true_result = sum(cm_all_true) / len(cm_all_true)
        #
        v_images = v_images.to(device=device, dtype=torch.float32)
        #
        v_outputs_logits_original, v_outputs_logits_noisy = model(v_images)
        #
        b, c, h, w = v_outputs_logits_original.size()
        #
        v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
        #
        _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)
        #
        save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_seg.png'
        save_name_label = save_path + '/test_' + imagename[0] + '_' + str(i) + '_label.png'
        #
        bb, cc, hh, ww = v_images.size()
        #
        for ccc in range(cc):
            #
            save_name_slice = save_path + '/test_' + imagename[0] + '_' + str(i) + '_slice_' + str(ccc) + '.png'
            plt.imsave(save_name_slice, v_images[:, ccc, :, :].reshape(h, w).cpu().detach().numpy(), cmap='gray')
        #
        if class_no == 2:
            #
            plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            plt.imsave(save_name_label, labels_good.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            #
        else:
            testoutput_original = np.asarray(v_outputs_logits.cpu().detach().numpy(), dtype=np.uint8)
            testoutput_original = np.squeeze(testoutput_original, axis=0)
            testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)
            segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
            segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
            segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #
            segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
            segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            #
            segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
            imageio.imsave(save_name, segmentation_map)
            #
            testoutput_original = np.asarray(labels_good.reshape(h, w).cpu().detach().numpy(), dtype=np.uint8)
            testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)
            segmentation_map = np.zeros((h, w, 3), dtype=np.uint8)
            segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 255
            segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 1, testoutput_original[:, :, 1] == 1, testoutput_original[:, :, 2] == 1)] = 0
            #
            segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 255
            segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 2, testoutput_original[:, :, 1] == 2, testoutput_original[:, :, 2] == 2)] = 0
            #
            segmentation_map[:, :, 0][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            segmentation_map[:, :, 1][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 0
            segmentation_map[:, :, 2][np.logical_and(testoutput_original[:, :, 0] == 3, testoutput_original[:, :, 1] == 3, testoutput_original[:, :, 2] == 3)] = 255
            imageio.imsave(save_name_label, segmentation_map)
            #

if __name__ == '__main__':
    #
    # This is to run segmentation of trained our models
    # To use it, change:
    #   1. model_path
    #   2. test_pat: path to testing data
    model_path = '../../saved_model.pt'
    # data path:
    test_path = '../../data_folder'
    dataset_tag = 'brats'
    label_mode = 'multi'
    test_data = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)
    segmentation(model_name='Unet_CMs', model_path=model_path, testdata=test_data, class_no=4, data_set=dataset_tag)




