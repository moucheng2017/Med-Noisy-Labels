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
                #labels_CR = labels_CR.to(device = device, dtype = torch.float32)

                labels_all = []
                labels_all.append(labels_AR)
                labels_all.append(labels_HS)
                labels_all.append(labels_SG)
                #labels_all.append(labels_CR)

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

                    v_dice, v_ged = evaluate_noisy_label_4(data = validateloader,
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

                    # Include Writer
    #
    model.eval()

    # save model
    # ==========
    save_path = path_name + '/Exp_Results_CMs_step'

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
    save_path = path_name + '/Exp_Results_CMs_step/' + dataset_tag
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

    ### =============== ###
    ### === TESTING === ###
    ### =============== ###
    

    if dataset_tag == 'oocytes_gent':

        for i, (v_images, labels_AR, labels_HS, labels_SG, labels_CR, imagename) in enumerate(testdata):

            cm_all_true = []
            cm_mse = 0                  # Mean Squared Error

            cm_AR_true = calculate_cm(pred = labels_AR, true = labels_CR)
            cm_HS_true = calculate_cm(pred = labels_HS, true = labels_CR)
            cm_SG_true = calculate_cm(pred = labels_SG, true = labels_CR)

            cm_all_true.append(cm_AR_true)
            cm_all_true.append(cm_HS_true)
            cm_all_true.append(cm_SG_true)

            v_images = v_images.to(device = device, dtype = torch.float32)

            v_outputs_logits_original, v_outputs_logits_noisy = model(v_images)

            b, c, h, w = v_outputs_logits_original.size()
            print("Output logits size:", v_outputs_logits_original.size())

            v_outputs_logits_original = nn.Softmax(dim = 1)(v_outputs_logits_original)

            _, v_outputs_logits = torch.max(v_outputs_logits_original, dim = 1)

            # save #
            # ---- #
            save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_cms_step.png'
            save_name_label = save_path + '/test_' + imagename[0] + '_' + str(i) + '_label.png'
            #
            plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
            plt.imsave(save_name_label, labels_CR.reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
            #
            bb, cc, hh, ww = v_images.size()
            for ccc in range(cc):
                #
                save_name_slice = save_path + '/test_' + imagename[0] + '_' + str(i) + '_slice_' + str(ccc) + '.png'
                plt.imsave(save_name_slice, v_images[:, ccc, :, :].reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
                #
            if save_probability_map is True:
                for class_index in range(c):
                    #
                    if c > 0:
                        v_outputs_logits = v_outputs_logits_original[:, class_index, :, :]
                        save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_class_' + str(class_index) + '_cms_step_probability.png'
                        plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
            # ---- #

            v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h * w)
            v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
            v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b * h * w, c, 1)

            for j, cm in enumerate(v_outputs_logits_noisy):
                
                if low_rank_mode is False:

                    cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
                    
                    cm = cm / cm.sum(1, keepdim = True)
                    
                    v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b * h * w, c)
                    
                    v_noisy_output_original = v_noisy_output_original.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
                    
                    if j < len(cm_all_true):
                        
                        cm_pred_ = cm.sum(0) / (b * h * w)
                        cm_pred_ = cm_pred_.cpu().detach().numpy()

                        cm_true_ = cm_all_true[j]
                        
                        # MSE
                        cm_mse_each_label = cm_pred_ - cm_true_
                        cm_mse_each_label = cm_mse_each_label ** 2

                        cm_mse += cm_mse_each_label.mean()

                        _, v_noisy_output = torch.max(v_noisy_output_original, dim = 1)

                # save #
                # ---- #
                save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_noisy_' + str(j) + '_cms_step.png'
                
                save_cm_name = save_path + '/' + imagename[0] + '_cm.npy'
                np.save(save_cm_name, cm.cpu().detach().numpy())
                
                print("CM shape:", cm.shape)

                plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
                
                if save_probability_map is True:
                    
                    for class_index in range(c):
                        
                        if c > 0:
                            
                            v_noisy_output = v_noisy_output_original[:, class_index, :, :]
                            save_name = save_path + '/test_' + imagename[0] + '_' + str(i) + '_noisy_class_' + str(class_index) + '_cms_step_probability.png'
                            plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap = 'gray')
                # ---- #

    ### =============== ###

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




    print('\nTraining finished and model saved\n')
    
    return model


