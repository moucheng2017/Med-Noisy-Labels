import os
import errno
import torch
import timeit

import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
import torch.functional as F
from torch.utils import data

from sklearn.metrics.pairwise import cosine_distances
from scipy import spatial
from sklearn.metrics import mean_squared_error
from torch.optim import lr_scheduler
from Loss import dice_loss
from Utilis import segmentation_scores
from Utilis import CustomDataset, evaluate, test
from tensorboardX import SummaryWriter
from torch.autograd import Variable

from Models import UNet


def trainUnet(dataset_tag,
              dataset_name,
              data_directory,
              input_dim,
              class_no,
              repeat,
              train_batchsize,
              validate_batchsize,
              num_epochs,
              learning_rate,
              width,
              depth,
              augmentation='all_flip',
              loss_f='dice'):

    """ This is the panel to control the training of baseline U-net.

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
        Exp = UNet(in_ch=input_dim,
                   width=width,
                   depth=depth,
                   class_no=class_no,
                   norm='in',
                   dropout=False,
                   apply_last_layer=True)
        #
        Exp_name = 'UNet' + '_width' + str(width) + \
                   '_depth' + str(depth) + \
                   '_repeat' + str(j)
        #
        # ====================================================================================================================================================================
        trainloader, validateloader, testloader, data_length = getData(data_directory, dataset_name, dataset_tag, train_batchsize, validate_batchsize, augmentation)
        # ==================
        trainSingleModel(Exp,
                         Exp_name,
                         num_epochs,
                         data_length,
                         learning_rate,
                         dataset_tag,
                         train_batchsize,
                         trainloader,
                         validateloader,
                         testloader,
                         losstag=loss_f,
                         class_no=class_no)


def getData(data_directory, dataset_name, dataset_tag, train_batchsize, validate_batchsize, data_augment):
    #
    train_image_folder = data_directory + '/' + dataset_name + '/' + \
        dataset_tag + '/train/patches'
    train_label_folder = data_directory + '/' + dataset_name + '/' + \
        dataset_tag + '/train/labels'
    validate_image_folder = data_directory + '/' + dataset_name + '/' + \
        dataset_tag + '/validate/patches'
    validate_label_folder = data_directory + '/' + dataset_name + '/' + \
        dataset_tag + '/validate/labels'
    test_image_folder = data_directory + '/' + dataset_name + '/' + \
        dataset_tag + '/test/patches'
    test_label_folder = data_directory + '/' + dataset_name + '/' + \
        dataset_tag + '/test/labels'
    #
    train_dataset = CustomDataset(train_image_folder, train_label_folder, data_augment)
    #
    validate_dataset = CustomDataset(validate_image_folder, validate_label_folder, data_augment)
    #
    test_dataset = CustomDataset(test_image_folder, test_label_folder, 'none')
    #
    trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=5, drop_last=True)
    #
    validateloader = data.DataLoader(validate_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False)
    #
    testloader = data.DataLoader(test_dataset, batch_size=validate_batchsize, shuffle=False, num_workers=validate_batchsize, drop_last=False)
    #
    return trainloader, validateloader, testloader, len(train_dataset)

# =====================================================================================================================================


def trainSingleModel(model,
                     model_name,
                     num_epochs,
                     data_length,
                     learning_rate,
                     datasettag,
                     train_batchsize,
                     trainloader,
                     validateloader,
                     testdata,
                     losstag,
                     class_no):
    # change log names
    iteration_amount = data_length // train_batchsize - 1
    #
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #
    lr_str = str(learning_rate)
    #
    epoches_str = str(num_epochs)
    #
    save_model_name = model_name + '_' + datasettag + '_e' + epoches_str + '_lr' + lr_str
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
    #
    print('The current model is:')
    #
    print(save_model_name)
    #
    print('\n')
    #
    writer = SummaryWriter('../../Results/Log_' + datasettag + '/' + save_model_name)

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):
        #
        model.train()
        running_loss = 0
        running_iou = 0
        #
        for j, (images, labels, imagename) in enumerate(trainloader):
            # check label values:
            # unique, counts = np.unique(labels, return_counts=True)
            # print(np.asarray((unique, counts)).T)
            #
            optimizer.zero_grad()
            #
            images = images.to(device=device, dtype=torch.float32)

            if class_no == 2:

                labels = labels.to(device=device, dtype=torch.float32)

            else:

                labels = labels.to(device=device, dtype=torch.long)

            outputs_logits = model(images)
            #
            if class_no == 2:
                #
                if losstag == 'dice':
                    #
                    loss = dice_loss(torch.sigmoid(outputs_logits), labels)
                    #
                elif losstag == 'ce':
                    #
                    loss = nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels)
                    #
                elif losstag == 'hybrid':
                    #
                    loss = dice_loss(torch.sigmoid(outputs_logits), labels) + nn.BCEWithLogitsLoss(reduction='mean')(outputs_logits, labels)
            #
            else:
                #
                loss = nn.CrossEntropyLoss(reduction='mean')(torch.softmax(outputs_logits, dim=1), labels.squeeze(1))
                #
            loss.backward()
            optimizer.step()
            #
            if class_no == 2:
                outputs_logits = torch.sigmoid(outputs_logits)
            else:
                outputs_logits = torch.softmax(outputs_logits, dim=1)
            #
            train_iou = segmentation_scores(labels.cpu().detach().numpy(), outputs_logits.cpu().detach().numpy(), class_no)
            running_loss += loss
            running_iou += train_iou
            #
            if (j + 1) % iteration_amount == 0:
                #
                validate_iou = evaluate(validateloader, model, device, class_no=class_no)
                print(
                    'Step [{}/{}], Train loss: {:.4f}, '
                    'Train iou: {:.4f}, '
                    'Val iou: {:.4f}, '.format(epoch + 1, num_epochs,
                                               running_loss / (j + 1),
                                               running_iou / (j + 1),
                                               validate_iou))
                # # # ================================================================== #
                # # #                        TensorboardX Logging                        #
                # # # # ================================================================ #
                writer.add_scalars('scalars', {'loss': running_loss / (j + 1),
                                               'train iou': running_iou / (j + 1),
                                               'val iou': validate_iou}, epoch + 1)
                #
        for param_group in optimizer.param_groups:
            param_group['lr'] = learning_rate*((1 - epoch / num_epochs)**0.999)
            #
    test(testdata,
         model,
         device,
         class_no=class_no,
         save_path=saved_information_path)
    #
    # save model
    stop = timeit.default_timer()
    #
    print('Time: ', stop - start)
    #
    save_model_name_full = saved_model_path + '/' + save_model_name + '_Final.pt'
    #
    path_model = save_model_name_full
    #
    torch.save(model, path_model)
    #
    print('\nTraining finished and model saved\n')
    #
    return model

