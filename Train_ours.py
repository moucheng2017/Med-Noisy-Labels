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


def trainModels(input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                alpha,
                width,
                depth,
                data_path,
                dataset_tag,
                label_mode,
                loss_f='noisy_label',
                save_probability_map=True,
                low_rank_mode=False):

    """ This is the panel to control the hyper-parameter of training of our methods.

    Args:
        input_dim: channel number of input image, for example, 3 for RGB
        class_no: number of classes of classification
        repeat: repat the same experiments with different stochastic seeds, we normally run each experiment at least 3 times
        train_batchsize: training batch size, this depends on the GPU memory
        validate_batchsize: we normally set-up as 1
        num_epochs: training epoch length
        learning_rate:
        input_height: resolution of input image
        input_width: resolution of input image
        alpha: regularisation strength hyper-parameter
        width: channel number of first encoder in the segmentation network, for the standard U-net, it is 64
        depth: down-sampling stages of the segmentation network
        data_path: path to where you store your all of your data
        dataset_tag: 'mnist' for MNIST; 'brats' for BRATS 2018; 'lidc' for LIDC lung data set
        label_mode: 'multi' for multi-class of proposed method; 'p_unet' for baseline probabilistic u-net; 'normal' for binary on MNIST; 'binary' for general binary segmentation
        loss_f: 'noisy_label' for our noisy label function, or 'dice' for dice loss
        save_probability_map: if True, we save all of the probability maps of output of networks

    Returns:

    """
    for j in range(1, repeat + 1):
        #
        Segmentation_net = UNet_CMs(in_ch=input_dim, width=width, depth=depth, class_no=class_no, norm='in', low_rank=low_rank_mode)
        Exp_name = 'Seg_UNet_CMs_Direct_' + '_width' + str(width) + \
                   '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                   '_repeat' + str(j) + '_alpha_' + str(alpha) + '_e' + str(num_epochs) + \
                   '_lr' + str(learning_rate) + '_save_probability_' + str(save_probability_map) + \
                   '_low_rank_mode_' + str(low_rank_mode)

        #
        # ====================================================================================================================================================================
        trainloader, validateloader, testloader, data_length = getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode)
        # ================================
        trainSingleModel(Segmentation_net,
                         Exp_name,
                         num_epochs,
                         data_length,
                         learning_rate,
                         alpha,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         losstag=loss_f,
                         class_no=class_no,
                         data_set=dataset_tag,
                         save_probability_map=save_probability_map,
                         low_rank_mode=low_rank_mode)


def getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode):
    #
    # train_image_folder = data_directory + '/' + dataset_name + '/' + \
    #     dataset_tag + '/train/patches'
    # train_label_folder = data_directory + '/' + dataset_name + '/' + \
    #     dataset_tag + '/train/labels'
    # validate_image_folder = data_directory + '/' + dataset_name + '/' + \
    #     dataset_tag + '/validate/patches'
    # validate_label_folder = data_directory + '/' + dataset_name + '/' + \
    #     dataset_tag + '/validate/labels'
    # test_image_folder = data_directory + '/' + dataset_name + '/' + \
    #     dataset_tag + '/test/patches'
    # test_label_folder = data_directory + '/' + dataset_name + '/' + \
    #     dataset_tag + '/test/labels'
    #
    # dataset_tag = 'mnist
    # noisylabel= 'multi
    #
    train_path = data_path + '/train'
    validate_path = data_path + '/validate'
    test_path = data_path + '/test'
    #
    train_dataset = CustomDataset_punet(dataset_location=train_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=True)

    validate_dataset = CustomDataset_punet(dataset_location=validate_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)

    test_dataset = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)
    #
    # train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment)
    # #
    # validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, data_augment)
    # #
    # test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none')
    #
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=5, drop_last=True)
    #
    validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False)
    #
    testloader = data.DataLoader(test_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False)
    #
    return trainloader, validateloader, testloader, len(train_dataset)

# =====================================================================================================================================


def trainSingleModel(model_seg,
                     model_name,
                     num_epochs,
                     data_length,
                     learning_rate,
                     alpha,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     losstag,
                     class_no,
                     data_set,
                     save_probability_map,
                     low_rank_mode):
    #
    # change log names
    iteration_amount = data_length // train_batchsize - 1
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    save_model_name = model_name
    #
    saved_information_path = '../../Results'
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    #
    saved_information_path = saved_information_path + '/Results_' + save_model_name
    #
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    saved_model_path = saved_information_path + '/trained_models'
    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter('../../Results/Log/Log_' + model_name)

    model_seg.to(device)
    # model_cm.to(device)

    optimizer1 = AdamW(model_seg.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=2e-5)
    # optimizer2 = torch.optim.Adam(model_cm.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):
        #
        model_seg.train()
        # model_cm.train()
        running_loss = 0
        running_loss_ce = 0
        running_loss_trace = 0
        running_iou = 0
        #
        if data_set == 'mnist' or data_set == 'brats':
            #
            for j, (images, labels_over, labels_under, labels_wrong, labels_good, imagename) in enumerate(trainloader):
                #
                b, c, h, w = images.size()
                #
                #
                optimizer1.zero_grad()
                # optimizer2.zero_grad()
                images = images.to(device=device, dtype=torch.float32)
                #
                labels_over = labels_over.to(device=device, dtype=torch.float32)
                labels_under = labels_under.to(device=device, dtype=torch.float32)
                labels_wrong = labels_wrong.to(device=device, dtype=torch.float32)
                labels_good = labels_good.to(device=device, dtype=torch.float32)
                #
                labels_all = []
                #
                labels_all.append(labels_over)
                labels_all.append(labels_under)
                labels_all.append(labels_wrong)
                labels_all.append(labels_good)
                #
                outputs_logits, outputs_logits_noisy = model_seg(images)
                #
                if low_rank_mode is False:
                    #
                    loss, loss_ce, loss_trace = noisy_label_loss(outputs_logits, outputs_logits_noisy, labels_all, alpha)
                    #
                else:
                    #
                    loss, loss_ce, loss_trace = noisy_label_loss_low_rank(outputs_logits, outputs_logits_noisy, labels_all, alpha)
                    #
                loss.backward()
                optimizer1.step()
                # optimizer2.step()
                #
                _, train_output = torch.max(outputs_logits, dim=1)
                #
                train_iou = segmentation_scores(labels_good.cpu().detach().numpy(), train_output.cpu().detach().numpy(), class_no)
                #
                # print(train_iou)
                # train_iou = segmentation_scores(labels_true.cpu().detach().numpy(), torch.sigmoid(outputs_logits[:, 0, :, :]).cpu().detach().numpy(), class_no)
                running_loss += loss
                running_loss_ce += loss_ce
                running_loss_trace += loss_trace
                running_iou += train_iou
                #
                # if (j + 1) % iteration_amount == 0:
                if (j + 1) == 1:
                    #
                    if low_rank_mode is False:
                        v_dice, v_ged = evaluate_noisy_label_4(data=validateloader,
                                                               model1=model_seg,
                                                               class_no=class_no)
                    else:
                        v_dice, v_ged = evaluate_noisy_label_6(data=validateloader,
                                                               model1=model_seg,
                                                               class_no=class_no)
                    #
                    print(
                        'Step [{}/{}], '
                        'Train loss: {:.4f}, '
                        'Train dice: {:.4f},'
                        'Validate dice: {:.4f},'
                        'Validate GED: {:.4f},'
                        'Train loss main: {:.4f},'
                        'Train loss regualrisation: {:.4f},'.format(epoch + 1, num_epochs,
                                                                    running_loss / (j + 1),
                                                                    running_iou / (j + 1),
                                                                    v_dice,
                                                                    v_ged,
                                                                    running_loss_ce / (j + 1),
                                                                    running_loss_trace / (j + 1)))
                #
                    writer.add_scalars('scalars', {'loss': running_loss / (j + 1),
                                                   'train iou': running_iou / (j + 1),
                                                   'val iou': v_dice,
                                                   'train main loss': running_loss_ce / (j + 1),
                                                   'train regularisation loss': running_loss_trace / (j + 1)}, epoch + 1)
                #
        elif data_set == 'lidc':
            #
            for j, (images, labels_over, labels_under, labels_wrong, labels_good, labels_true, imagename) in enumerate(trainloader):
                #
                b, c, h, w = images.size()
                #
                #
                optimizer1.zero_grad()
                # optimizer2.zero_grad()
                images = images.to(device=device, dtype=torch.float32)
                #
                labels_over = labels_over.to(device=device, dtype=torch.float32)
                labels_under = labels_under.to(device=device, dtype=torch.float32)
                labels_wrong = labels_wrong.to(device=device, dtype=torch.float32)
                labels_good = labels_good.to(device=device, dtype=torch.float32)
                labels_true = labels_true.to(device=device, dtype=torch.float32)
                #
                labels_all = []
                #
                labels_all.append(labels_over)
                labels_all.append(labels_under)
                labels_all.append(labels_wrong)
                labels_all.append(labels_good)
                #
                outputs_logits, outputs_logits_noisy = model_seg(images)
                #
                loss, loss_ce, loss_trace = noisy_label_loss(outputs_logits, outputs_logits_noisy, labels_all, alpha)

                loss.backward()
                optimizer1.step()
                # optimizer2.step()
                #
                _, train_output = torch.max(outputs_logits, dim=1)
                #
                train_iou = segmentation_scores(labels_true.cpu().detach().numpy(), train_output.cpu().detach().numpy(), class_no)
                #
                # print(train_iou)
                # train_iou = segmentation_scores(labels_true.cpu().detach().numpy(), torch.sigmoid(outputs_logits[:, 0, :, :]).cpu().detach().numpy(), class_no)
                running_loss += loss
                running_loss_ce += loss_ce
                running_loss_trace += loss_trace
                running_iou += train_iou
                #
                # if (j + 1) % iteration_amount == 0:
                if (j + 1) == 1:
                    #
                    #
                    v_dice, v_ged = evaluate_noisy_label_5(data=validateloader,
                                                           model1=model_seg,
                                                           class_no=class_no)
                    #
                    print(
                        'Step [{}/{}], '
                        'Train loss: {:.4f}, '
                        'Train dice: {:.4f},'
                        'Validate dice: {:.4f},'
                        'Validate GED: {:.4f},'
                        'Train loss main: {:.4f},'
                        'Train loss regualrisation: {:.4f},'.format(epoch + 1, num_epochs,
                                                                    running_loss / (j + 1),
                                                                    running_iou / (j + 1),
                                                                    v_dice,
                                                                    v_ged,
                                                                    running_loss_ce / (j + 1),
                                                                    running_loss_trace / (j + 1)))
                    #
                    writer.add_scalars('scalars', {'loss': running_loss / (j + 1),
                                                   'train iou': running_iou / (j + 1),
                                                   'val iou': v_dice,
                                                   'train main loss': running_loss_ce / (j + 1),
                                                   'train regularisation loss': running_loss_trace / (j + 1)}, epoch + 1)
                    #
                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #

        # for param_group in optimizer.param_groups:
        #     param_group['lr'] = learning_rate*((1 - epoch / num_epochs)**0.999)
        #
    model_seg.eval()
    # model_cm.eval()
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
                '_Noisy_Label_Net_' + save_model_name
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
    if data_set == 'mnist' or data_set == 'brats':
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
            v_outputs_logits_original, v_outputs_logits_noisy = model_seg(v_images)
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
            if save_probability_map is True:
                for class_index in range(c):
                    #
                    if c > 0:
                        v_outputs_logits = v_outputs_logits_original[:, class_index, :, :]
                        save_name = save_path + '/test_' + imagename[0] + str(i) + '_class_' + str(class_index) + '_seg_probability.png'
                        plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            #
            nnn = 1
            #
            v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h*w)
            v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
            v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)
            #
            cm_mse = 0
            #
            for j, cm in enumerate(v_outputs_logits_noisy):
                #
                if low_rank_mode is False:
                    #
                    cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
                    cm = cm / cm.sum(1, keepdim=True)
                    #
                    if j < len(cm_all_true):
                        #
                        cm_pred_ = cm.sum(0) / (b*h*w)
                        #
                        # print(np.shape(cm_pred_))
                        #
                        cm_pred_ = cm_pred_.cpu().detach().numpy()
                        #
                        # print(np.shape(cm_pred_))
                        #
                        cm_true_ = cm_all_true[j]
                        #
                        # print(np.shape(cm_true_))
                        #
                        cm_mse_each_label = cm_pred_ - cm_true_
                        #
                        cm_mse_each_label = cm_mse_each_label**2
                        # cm_mse_each_label = (cm.cpu().detach().numpy - cm_all_true[j])**2
                        cm_mse += cm_mse_each_label.mean()
                        #
                        # print(cm_mse)
                    #
                    v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b*h*w, c)
                    v_noisy_output_original = v_noisy_output_original.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                    #
                else:
                    #
                    b, c_r_d, h, w = cm.size()
                    r = c_r_d // c // 2
                    cm1 = cm[:, 0:r * c, :, :]
                    # cm1: b x c*rank x h x w
                    if r == 1:
                        cm2 = cm[:, r * c:c_r_d-1, :, :]
                    else:
                        cm2 = cm[:, r * c:c_r_d-1, :, :]
                    # cm2: b x c*rank x h x w
                    #
                    cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
                    # cm1: b*h*w x r x c
                    cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
                    #
                    cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
                    # cm1: b*h*w x r x c, normalisation along rows
                    cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
                    #
                    v_noisy_output_original = torch.bmm(cm1_reshape, v_outputs_logits_original)
                    # pred_noisy: b*h*w x r x 1
                    v_noisy_output_original = torch.bmm(cm2_reshape, v_noisy_output_original).view(b * h * w, c)
                    # pred_noisy: b*h*w x c x 1
                    v_noisy_output_original = v_noisy_output_original.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                    #
                #
                _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
                # print('noisy ' + str(nnn) + ' of test ' + str(i))
                # print(torch.sum(cm, dim=0) / (b * h * w))
                nnn += 1
                # print('\n')
                save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_noisy_' + str(j) + '_seg.png'
                #
                save_cm_name = save_path + '/' + imagename[0] + '_cm.npy'
                np.save(save_cm_name, cm.cpu().detach().numpy())
                #
                if class_no == 2:
                    #
                    plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')
                    #
                else:
                    #
                    testoutput_original = np.asarray(v_noisy_output.cpu().detach().numpy(), dtype=np.uint8)
                    testoutput_original = np.squeeze(testoutput_original, axis=0)
                    testoutput_original = np.repeat(testoutput_original[:, :, np.newaxis], 3, axis=2)
                    #
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
                if save_probability_map is True:
                    #
                    for class_index in range(c):
                        #
                        if c > 0:
                            #
                            v_noisy_output = v_noisy_output_original[:, class_index, :, :]
                            save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_noisy_class_' + str(class_index) + '_seg_probability.png'
                            plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')
                            #
    elif data_set == 'lidc':
        #
        for i, (v_images, labels_over, labels_under, labels_wrong, labels_good, labels_true, imagename) in enumerate(testdata):
            #
            cm_all_true = []
            cm_mse = 0
            #
            cm_over_true = calculate_cm(pred=labels_over, true=labels_true)
            cm_under_true = calculate_cm(pred=labels_under, true=labels_true)
            cm_wrong_true = calculate_cm(pred=labels_wrong, true=labels_true)
            cm_good_true = calculate_cm(pred=labels_good, true=labels_true)
            #
            cm_all_true.append(cm_over_true)
            cm_all_true.append(cm_under_true)
            cm_all_true.append(cm_wrong_true)
            cm_all_true.append(cm_good_true)
            #
            v_images = v_images.to(device=device, dtype=torch.float32)
            #
            v_outputs_logits_original, v_outputs_logits_noisy = model_seg(v_images)
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
            plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            plt.imsave(save_name_label, labels_true.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            #
            bb, cc, hh, ww = v_images.size()
            for ccc in range(cc):
                #
                save_name_slice = save_path + '/test_' + imagename[0] + '_' + str(i) + '_slice_' + str(ccc) + '.png'
                plt.imsave(save_name_slice, v_images[:, ccc, :, :].reshape(h, w).cpu().detach().numpy(), cmap='gray')
                #
            if save_probability_map is True:
                for class_index in range(c):
                    #
                    if c > 0:
                        v_outputs_logits = v_outputs_logits_original[:, class_index, :, :]
                        save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_class_' + str(class_index) + '_seg_probability.png'
                        plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
            #
            nnn = 1
            #
            v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h*w)
            v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
            v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)
            #
            for j, cm in enumerate(v_outputs_logits_noisy):
                #
                if low_rank_mode is False:
                    cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
                    #
                    cm = cm / cm.sum(1, keepdim=True)
                    #
                    v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b*h*w, c)
                    #
                    v_noisy_output_original = v_noisy_output_original.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                    #
                    if j < len(cm_all_true):
                        #
                        cm_pred_ = cm.sum(0) / (b*h*w)
                        #
                        # print(np.shape(cm_pred_))
                        #
                        cm_pred_ = cm_pred_.cpu().detach().numpy()
                        #
                        # print(np.shape(cm_pred_))
                        #
                        cm_true_ = cm_all_true[j]
                        #
                        # print(np.shape(cm_true_))
                        #
                        cm_mse_each_label = cm_pred_ - cm_true_
                        #
                        cm_mse_each_label = cm_mse_each_label**2
                        # cm_mse_each_label = (cm.cpu().detach().numpy - cm_all_true[j])**2
                        cm_mse += cm_mse_each_label.mean()
                        #
                        # print(cm_mse)
                    #
                else:
                    b, c_r_d, h, w = cm.size()
                    r = c_r_d // c // 2
                    cm1 = cm[:, 0:r * c, :, :]
                    # cm1: b x c*rank x h x w
                    cm2 = cm[:, r * c:c_r_d, :, :]
                    # cm2: b x c*rank x h x w
                    #
                    cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
                    # cm1: b*h*w x r x c
                    cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
                    #
                    cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
                    # cm1: b*h*w x r x c, normalisation along rows
                    cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
                    #
                    v_noisy_output_original = torch.bmm(cm1_reshape, v_outputs_logits_original)
                    # pred_noisy: b*h*w x r x 1
                    v_noisy_output_original = torch.bmm(cm2_reshape, v_noisy_output_original).view(b * h * w, c)
                    # pred_noisy: b*h*w x c x 1
                    v_noisy_output_original = v_noisy_output_original.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                #
                _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
                #
                # print('noisy ' + str(nnn) + ' of test ' + str(i))
                # print(torch.sum(cm, dim=0) / (b * h * w))
                # nnn += 1
                # print('\n')
                #
                save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_noisy_' + str(j) + '_seg.png'
                #
                save_cm_name = save_path + '/' + imagename[0] + '_cm.npy'
                np.save(save_cm_name, cm.cpu().detach().numpy())
                #
                plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')
                #
                if save_probability_map is True:
                    #
                    for class_index in range(c):
                        #
                        if c > 0:
                            #
                            v_noisy_output = v_noisy_output_original[:, class_index, :, :]
                            save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_noisy_class_' + str(class_index) + '_seg_probability.png'
                            plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')
                            #
    # save model
    stop = timeit.default_timer()
    #
    print('Time: ', stop - start)
    #
    save_model_name_full = saved_model_path + '/' + save_model_name + '_Final_seg.pt'
    #
    path_model = save_model_name_full
    #
    torch.save(model_seg, path_model)
    #
    # save_model_name_full = saved_model_path + '/' + save_model_name + '_Final_cm.pt'
    #
    # path_model = save_model_name_full
    #
    # torch.save(model_cm, path_model)
    #
    result_dictionary = {'Test Dice': str(v_dice), 'Test GED': str(v_ged), 'Test CM MSE': str(cm_mse / (i + 1))}
    ff_path = save_path + '/test_result_data.txt'
    ff = open(ff_path, 'w')
    ff.write(str(result_dictionary))
    ff.close()
    #
    print('\nTraining finished and model saved\n')
    #
    return model_seg

