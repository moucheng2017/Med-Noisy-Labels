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
from Loss import noisy_label_loss_low_rank, noisy_label_loss, cm_loss
from Models import UNet_GlobalCMs, CMNet

from Utilis import evaluate_noisy_label_4, evaluate_noisy_label_5, evaluate_noisy_label_6

def trainStepCM(input_dim,
                class_no,
                repeat,
                train_batchsize,
                validate_batchsize,
                num_epochs,
                learning_rate,
                input_height,
                input_width,
                alpha,
                width,
                depth,
                data_path,
                dataset_tag,
                label_mode,
                loss_f = 'noisy_label',
                save_probability_map = True,
                path_name = './Results'):

    """ info

    Args:
        input_dim:
        ...
        ...
        ...

    Returns:

    """

    for j in range(1, repeat + 1):

        Regularization_net = CMNet(in_ch = input_dim,
                                   width = width,
                                   depth = depth,
                                   class_no = class_no,
                                   input_height = input_height,
                                   input_width = input_width,
                                   norm = 'in')

        Exp_name = 'Reg_CMNet' + '_width' + str(width) + \
                   '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
                   '_repeat' + str(j) + '_alpha_' + str(alpha) + '_e' + str(num_epochs) + \
                   '_lr' + str(learning_rate) + '_save_probability_' + str(save_probability_map)

        # load data
        trainloader, validateloader, testloader, data_length = getData(train_batchsize = train_batchsize,
                                                                       validate_batchsize = validate_batchsize,
                                                                       data_path = data_path,
                                                                       dataset_tag = dataset_tag,
                                                                       label_mode = label_mode)

        # train model
        trainModelCM(model = Regularization_net,
                     model_name = Exp_name,
                     num_epochs = num_epochs,
                     data_length = data_length,
                     learning_rate = learning_rate,
                     alpha = alpha,
                     train_batchsize = train_batchsize,
                     trainloader = trainloader,
                     validateloader = validateloader,
                     testdata = testloader,
                     losstag = loss_f,
                     class_no = class_no,
                     dataset_tag = dataset_tag,
                     save_probability_map = save_probability_map,
                     low_rank_mode = False,
                     path_name = path_name)



def getData(train_batchsize, validate_batchsize, data_path, dataset_tag, label_mode):

    train_path = data_path + '/train'
    validate_path = data_path + '/validate'
    test_path = data_path + '/test'

    train_dataset = CustomDataset_punet(dataset_location = train_path, dataset_tag = dataset_tag, noisylabel = label_mode, augmentation = True)
    validate_dataset = CustomDataset_punet(dataset_location = validate_path, dataset_tag = dataset_tag, noisylabel = label_mode, augmentation = False)
    test_dataset = CustomDataset_punet(dataset_location = test_path, dataset_tag = dataset_tag, noisylabel = label_mode, augmentation = False)

    trainloader = data.DataLoader(train_dataset, batch_size = train_batchsize, shuffle = True, num_workers = 5, drop_last = True)
    validateloader = data.DataLoader(validate_dataset, batch_size = validate_batchsize, shuffle = False, num_workers = validate_batchsize, drop_last = False)
    testloader = data.DataLoader(test_dataset, batch_size = validate_batchsize, shuffle = False, num_workers = validate_batchsize, drop_last = False)

    return trainloader, validateloader, testloader, len(train_dataset)

def trainModelCM(model,
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
                 dataset_tag,
                 save_probability_map,
                 low_rank_mode = False,
                 path_name = './Results'):

    iteration_amount = data_length // train_batchsize - 1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    save_model_name = model_name

    saved_information_path = path_name

    # create folders
    # --------------
    try:
        os.mkdir(saved_information_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    saved_model_path = saved_information_path + '/trained_models'

    try:
        os.mkdir(saved_model_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
        pass

    write = SummaryWriter(path_name + '/Log/Log_' + model_name)

    print("The current model is ", save_model_name)
    # --------------
    # model
    model = model.to(device)

    # optimizer
    # AdamW: same as it was given
    optimizer = AdamW(model.parameters(), lr = learning_rate, betas = (0.9, 0.999), eps = 1e-8, weight_decay = 2e-5)

    start = timeit.default_timer()

    for epoch in range(num_epochs):

        running_loss = 0.0
        running_loss_ce = 0.0
        running_loss_trace = 0.0
        running_iou = 0.0

        if dataset_tag == 'oocytes_gent':

            # labels:
            # AR: annotator 1
            # HS: annotator 2
            # SG: annotator 3
            # CR: control labels, used as point of reference / validate / etc.
            for j, (images, labels_AR, labels_HS, labels_SG, labels_CR, imagename) in enumerate(trainloader):

                # b: batch, c: channels, h: height, w: width
                b, c, h, w = images.size()

                # images
                images = images.to(device = device, dtype = torch.float32)

                # labels
                labels_AR = labels_AR.to(device = device, dtype = torch.float32)
                labels_HS = labels_HS.to(device = device, dtype = torch.float32)
                labels_SG = labels_SG.to(device = device, dtype = torch.float32)
                labels_CR = labels_CR.to(device = device, dtype = torch.float32)

                labels_all = []
                labels_all.append(labels_AR)
                labels_all.append(labels_HS)
                labels_all.append(labels_SG)
                labels_all.append(labels_CR)

                # model
                y_init, y_cms = model(images)

                # loss
                loss, loss_ce, loss_trace = cm_loss(y_init, y_cms, labels_all, alpha)

                loss.backward()
                optimizer.step()

                _, train_output = torch.max(y_init, dim = 1)

                # (I)ntersection (o)ver (U)nion
                train_iou = segmentation_scores(labels_CR.cpu().detach().numpy(),
                                                train_output.cpu().detach().numpy(),
                                                class_no)

                # Update losses & IoU
                running_loss += loss
                running_loss_ce += loss_ce
                running_loss_trace += loss_trace
                running_iou += train_iou

                if (j + 1) == 1:

                    v_dice, v_ged = evaluate_noisy_label_6(data = validateloader,
                                                           model1 = model,
                                                           class_no = class_no)

                    print(
                            'Step [{}/{}], '
                            'Train loss: {:.4f}, '
                            'Train dice: {:.4f},'
                            'Validate dice: {:.4f},'
                            'Validate GED: {:.4f},'
                            'Train loss main: {:.4f},'
                            'Train loss regualrisation: {:.4f},'.format(epoch + 1, 
                                                                        num_epochs,
                                                                        running_loss / (j + 1),
                                                                        running_iou / (j + 1),
                                                                        v_dice,
                                                                        v_ged,
                                                                        running_loss_ce / (j + 1),
                                                                        running_loss_trace / (j + 1))
                            )






