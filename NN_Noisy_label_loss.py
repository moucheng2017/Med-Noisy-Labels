import torch
import torch.nn as nn

import torch.nn.functional as F
import matplotlib.pyplot as plt

from torch.autograd import Variable

import torch.autograd.grad_mode
torch.backends.cudnn.deterministic = True
from NNLoss import dice_loss
# ==========================


def noisy_label_loss_v13(pred, cms, labels, alpha, epoch, epoch_threshold, alpha_initial, regularisation_type):
    #
    if epoch < epoch_threshold:
        #
        beta_current = 0.0
        alpha_current = alpha_initial
        #
    else:
        #
        beta_current = 1.0
        alpha_current = alpha
    #
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    pred_norm = nn.Softmax(dim=1)(pred)
    pred_norm = pred_norm.view(b, c, h*w)
    pred_norm = pred_norm.permute(0, 2, 1).contiguous()
    pred_norm = pred_norm.view(b*h*w, c)
    pred_norm = pred_norm.view(b*h*w, c, 1)
    #
    for j, (cm, label_noisy) in enumerate(zip(cms, labels)):
        #
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        cm = cm / cm.sum(1, keepdim=True)
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        loss_current = beta_current*loss_current
        #
        if regularisation_type == '1':
            regularisation += torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w)
        elif regularisation_type == '2':
            regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
        elif regularisation_type == '3':
            regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
            regularisation += torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w)
        #
        main_loss += loss_current
        #
    regularisation = alpha_current*regularisation
    #
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def noisy_label_loss_low_rank(pred, cms, labels, alpha):
    # pred: prediction for true segmentation
    # cms: confusion matrices for each annotators
    # alpha: weight for trace
    main_loss = 0.0
    #
    regularisation = 0.0
    b, c, h, w = pred.size()
    # pred: b x c x h x w
    pred_norm = nn.Softmax(dim=1)(pred)
    # pred_norm: b x c x h x w
    pred_norm = pred_norm.view(b, c, h*w)
    # pred_norm: b x c x h*w
    pred_norm = pred_norm.permute(0, 2, 1).contiguous()
    # pred_norm: b x h*w x c
    pred_norm = pred_norm.view(b*h*w, c)
    # pred_norm: b*h*w x c
    pred_norm = pred_norm.view(b*h*w, c, 1)
    # pred_norm: b*h*w x c x 1
    #
    for j, (cm, label_noisy) in enumerate(zip(cms, labels)):
        #
        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2
        #
        if r == 1:
            #
            # print(cm.size())
            cm1 = cm[:, 0:r * c, :, :]
            # print(cm1.size())
            cm2 = cm[:, r * c:c_r_d-1, :, :]
            # print(cm2.size())
            scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
            #
            cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            #
            cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)
            #
            identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
            #
            # cm_reconstruct_approx = cm_reconstruct + identity_residual
            cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
            #
            cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)
            pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
            pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            regularisation_old = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)
            #
        else:
            #
            cm1 = cm[:, 0:r * c, :, :]
            cm2 = cm[:, r * c:c_r_d, :, :]
            # Old:
            # cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            # # cm1: b*h*w x r x c
            # cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            # #
            # cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
            # # cm1: b*h*w x r x c, normalisation along rows
            # cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
            # #
            # pred_noisy = torch.bmm(cm1_reshape, pred_norm)
            # # pred_noisy: b*h*w x r x 1
            # pred_noisy = torch.bmm(cm2_reshape, pred_noisy).view(b * h * w, c)
            # # pred_noisy: b*h*w x c x 1
            # pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            # # pred_noisy: b x c x h x w
            # #
            # identity_residual = torch.cat(b * h * w * [torch.eye(r, r)]).reshape(b * h * w, r, r).to(device='cuda', dtype=torch.float32)
            # #
            # cm_reconstruct = torch.bmm(cm1_reshape, cm2_reshape) + identity_residual
            # #
            # regularisation_old = torch.trace(torch.transpose(torch.sum(cm_reconstruct, dim=0), 0, 1)).sum() / (b * h * w)
            # loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
            # New:
            #
            # cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            # cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            # #
            # cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)
            # #
            # # identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
            # #
            # # cm_reconstruct_approx = cm_reconstruct + identity_residual
            # cm_reconstruct_approx = cm_reconstruct
            # #
            # cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)
            # pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
            # pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            # #
            # regularisation_old = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)
            #
            # print(cm.size())
            cm1 = cm[:, 0:r * c, :, :]
            # print(cm1.size())
            cm2 = cm[:, r * c:c_r_d-1, :, :]
            # print(cm2.size())
            scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
            #
            cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
            cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
            #
            cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)
            #
            identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
            #
            # cm_reconstruct_approx = cm_reconstruct + identity_residual
            cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
            #
            cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)
            pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
            pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            #
            regularisation_old = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)
        #
        # print(cm1_reshape.size())
        # print(cm2_reshape.size())
        #
        # (todo) put it back later
        # cm_reconstruct_sum = torch.sum(cm_reconstruct, dim=0)

        #
        # print(torch.trace(cm_reconstruct_sum))
        #
        # print(cm_reconstruct[0, ...])
        # cm_reconstruct: b*h*w x r x r
        # print(cm_reconstruct[0, :, :])
        # print(cm_reconstruct[1, :, :])
        #

        # pred_noisy: b x c x h x w
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        #
        # if regularisation_type == '1':
        # cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)
        # print(cm_reconstruct[0, ...])
        # r x r
        # print(cm_reconstruct[0, :, :])
        #
        # print(cm_reconstruct.size())
        # type 1:
        # regularisation1 = torch.trace(torch.sum(cm_reconstruct, dim=0)).sum() / (b * h * w)
        # regularisation += regularisation1
        regularisation += regularisation_old
        # print(regularisation1)
        #
        # print(regularisation1[0, ...])
        #
        # type 2:
        # regularisation += torch.trace(torch.transpose(torch.sum(cm_reconstruct, dim=0), 0, 1)).sum() / (b * h * w)
        #
        # regularisation += regularisation2
        #
        # print(regularisation2)
        #
        main_loss += loss_current
        #
    regularisation = alpha*regularisation
    #
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def noisy_label_loss_low_rank_new(pred, cms, labels, alpha, epoch, epoch_threshold):
    # pred: prediction for true segmentation
    # cms: confusion matrices for each annotators
    # alpha: weight for trace
    #
    if epoch < epoch_threshold:
        #
        alpha_current = 0
        #
    else:
        #
        alpha_current = alpha
    #
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    # pred: b x c x h x w
    pred_norm = nn.Softmax(dim=1)(pred)
    # pred_norm: b x c x h x w
    pred_norm = pred_norm.view(b, c, h*w)
    # pred_norm: b x c x h*w
    pred_norm = pred_norm.permute(0, 2, 1).contiguous()
    # pred_norm: b x h*w x c
    pred_norm = pred_norm.view(b*h*w, c)
    # pred_norm: b*h*w x c
    pred_norm = pred_norm.view(b*h*w, c, 1)
    # pred_norm: b*h*w x c x 1
    #
    for j, (cm, label_noisy) in enumerate(zip(cms, labels)):
        # cm: b x c*r*2 x h x w
        #    c_r_d = c*rank*2
        #    r: rank
        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2
        cm1 = cm[:, 0:r*c, :, :]
        # cm1: b x c*rank x h x w
        cm2 = cm[:, r*c:c_r_d, :, :]
        # cm2: b x c*rank x h x w
        #
        cm1_reshape = cm1.view(b, c_r_d // 2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, r*c).view(b*h*w, r, c)
        # cm1: b*h*w x r x c
        cm2_reshape = cm2.view(b, c_r_d // 2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, r*c).view(b*h*w, c, r)
        # cm2: b*h*w x c x r
        #
        # cm_reconstruct_ = torch.bmm(cm2_reshape, cm1_reshape)
        # cm_reconstruct_ = cm_reconstruct_ / cm_reconstruct_.sum(1, keepdim=True)
        # cm_reconstruct: b*h*w x r x r
        # print(cm_reconstruct_[0, :, :])
        # print(cm_reconstruct_[1, :, :])
        #
        cm1_reshape = cm1_reshape / cm1_reshape.sum(1, keepdim=True)
        # cm1: b*h*w x r x c, normalisation along rows
        cm2_reshape = cm2_reshape / cm2_reshape.sum(1, keepdim=True)
        # cm2: b*h*w x c x r, normalisation along columns
        #

        # cm_reconstruct: b*h*w x r x r
        # print(cm_reconstruct[0, :, :])
        # print(cm_reconstruct[1, :, :])
        #
        pred_noisy = torch.bmm(cm1_reshape, pred_norm)
        # pred_noisy: b*h*w x r x 1
        pred_noisy = torch.bmm(cm2_reshape, pred_noisy).view(b*h*w, c)
        # pred_noisy: b*h*w x c x 1
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        # pred_noisy: b x c x h x w
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        #
        # if regularisation_type == '1':
        cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)
        #
        # print(cm_reconstruct[0, :, :])
        #
        # print(cm_reconstruct.size())
        # type 1:
        # regularisation += torch.trace(torch.sum(cm_reconstruct, dim=0)).sum() / (b * h * w)
        # type 2:
        # regularisation += torch.trace(torch.transpose(torch.sum(cm_reconstruct, dim=0), 0, 1)).sum() / (b * h * w)
        #
        regularisation += 1 / (torch.trace(torch.transpose(torch.sum(cm_reconstruct, dim=0), 0, 1)).sum() / (b * h * w))
        #
        main_loss += loss_current
        #
    regularisation = alpha_current*regularisation
    #
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def identity_multiply(pred):
    #
    b, c, h, w = pred.size()
    #
    cm = torch.cat(b * h * w * [torch.eye(c, c)]).reshape(b*h*w, c, c).cuda()
    cm = cm / cm.sum(1, keepdim=True)
    pred_norm = nn.Softmax(dim=1)(pred)
    pred_noisy = torch.bmm(cm, pred_norm.reshape(b * h * w, c, 1)).reshape(b, c, h, w)
    #
    return pred_noisy


def noisy_label_loss(pred, cms, labels, alpha=0.1):
    # pred: b x c x h x w
    # cms: num_annotators x b x c**2 x h x w
    # label: b x h x w
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    pred_norm = nn.Softmax(dim=1)(pred)
    # pred_norm: b x c x h x w
    # print(pred_norm.size())
    # print(pred_norm.size())
    pred_norm = pred_norm.permute(0, 2, 3, 1).contiguous()
    # print(pred_norm.size())
    # pred_norm: b x h x w x c
    pred_norm = pred_norm.reshape(b * h * w, c, 1)
    # print(pred_norm.size())
    #
    for cm, label_noisy in zip(cms, labels):
        # (todo) cm = cm.permuate(b, h, w, c**2)
        # cm: b x c**2 x h x w
        # print(cm.size())
        cm = cm.permute(0, 2, 3, 1).contiguous().view(b*h*w, c, c)
        # print(cm.size())
        # cm: b x h x w x c**2
        # cm = cm.reshape(b*h*w, c**2)
        # cm: b*h*w x c**2
        # cm = cm.reshape(b*h*w, c, c)
        # print(cm.size())
        # cm: b*h*h x c x c
        # (todo) normalise along 1st dim: cm / cm.sum(1, keepdim=True)
        cm = cm / cm.sum(1, keepdim=True)
        # print(cm[0, :, :])
        # cm: b*h*w x c x c
        # cm[k, j, i] = p(annotator = j | true label = i)
        # probability of annotator labels class j when the true label is i at pixel k.
        # (todo) pred_norm.permuate(b, c, h , w)
        pred_noisy = torch.bmm(cm, pred_norm)
        # print(pred_noisy.size())
        # pred_noisy: b*h*w x c x 1
        # pred_noisy = pred_noisy.reshape(b, h, w, c)
        # print(pred_noisy.size())
        # pred_noisy = pred_noisy.permute(0, 3, 1, 2).contiguous()
        # print(pred_noisy.size())
        # pred_noisy: b x c x h x w
        pred_noisy = pred_noisy.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
        # print(pred_noisy.shape)
        # (todo) bmm_result.reshape(b, h, w, c).permuate(b, c, h, w)
        # (todo) make all blank 8
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        main_loss += loss_current
        # cm: b*h*w x c x c
        # (todo)
        regularisation += (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w))
        # regularisation += (1 / (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w)))
        # torch.sum(cm, dim=0): c x c
        #
    regularisation = alpha*regularisation
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def noisy_label_loss_v12(pred, cms, labels, alpha):
    #
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    pred_norm = nn.Softmax(dim=1)(pred)
    pred_norm = pred_norm.view(b, c, h*w)
    pred_norm = pred_norm.permute(0, 2, 1).contiguous()
    pred_norm = pred_norm.view(b*h*w, c)
    pred_norm = pred_norm.view(b*h*w, c, 1)
    #
    for j, (cm, label_noisy) in enumerate(zip(cms, labels)):
        #
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        cm = cm / cm.sum(1, keepdim=True)
        #
        # print(cm[0, :, :])
        #
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        #
        # if regularisation_type == '1':
        # regularisation += (1 / (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w)))
        # regularisation += torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w)
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
        # elif regularisation_type == '2':
        #     # regularisation += 1 / (torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w))
        #     regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
        # else:
        #     # regularisation += (1 / (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w))) + 1 / (torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w))
        #     regularisation += torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w) + torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
        #     #
        main_loss += loss_current
        #
    regularisation = alpha*regularisation
    #
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def noisy_label_loss_v11(pred, cms, labels, alpha=0.1):
    #
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    pred_norm = nn.Softmax(dim=1)(pred)
    #
    for cm, label_noisy in zip(cms, labels):
        cm = cm.reshape(b*h*w, c, c)
        cm = cm / cm.sum(1, keepdim=True)
        pred_noisy = torch.bmm(cm, pred_norm.reshape(b*h*w, c, 1)).reshape(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.reshape(b, h, w).long())
        main_loss += loss_current
        # regularisation += (1 / (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w)))
        regularisation += (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w))
        # gradient = torch.autograd.grad(outputs=loss_current, inputs=pred_noisy, create_graph=True, only_inputs=True)[0]
        # regularisation += ((gradient.norm(2, dim=1) - 1) ** 2).mean()
        # regularisation += (torch.trace(torch.sum(cm, dim=0)).sum() / (b * h * w))
        #

    regularisation = alpha*regularisation
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


# def noisy_label_loss_v9(pred, preds_noisy, labels, alpha=0):
#     #
#     main_loss = 0.0
#     regularisation = 0.0
#     b, c, h, w = pred.size()
#     pred = nn.Softmax(dim=1)(pred)
#     #
#     for pred_noisy, label_noisy in zip(preds_noisy, labels):
#         #
#         # pred_noisy_softmax = nn.Softmax(dim=1)(pred_noisy)
#         # _, pred_noisy = torch.max(pred_noisy, dim=1)
#         #
#         pred_noisy = SoftArgmax2D()(pred_noisy)
#         #
#         confusion_matrix = torch.zeros(b * h * w, c, c).to(device='cuda', dtype=torch.float)
#         #
#         for indice, (gt, p) in enumerate(zip(label_noisy.view(-1), pred_noisy.view(-1))):
#             #
#             confusion_matrix[indice, gt.long(), p.long()] += 1.0
#             #
#         print(confusion_matrix[0, :, :])
#         confusion_matrix = confusion_matrix / confusion_matrix.sum(1, keepdim=True)
#         #
#         pred_noisy_refined = torch.bmm(confusion_matrix.reshape(b*h*w, c, c).float(), pred.reshape(b*h*w, c, 1)).reshape(b, c, h, w)
#         #
#         pred_noisy_refined = SoftArgmax2D()(pred_noisy_refined)
#         dice_loss_label = dice_loss(pred_noisy_refined.float(), label_noisy.float())
#         #
#         main_loss += dice_loss_label
#         # method 2 for confusion matrix:
#         # a very weird trick
#         # # pred_noisy = torch.argmax(pred_noisy, dim=1).flatten().float()
#         # #
#         # _, pred_noisy = torch.max(pred_noisy, dim=1)
#         # pred_noisy = pred_noisy.flatten().float()
#         # #
#         # pred_all = c * label_noisy.flatten() + pred_noisy
#         # cm = torch.bincount(pred_all.long(), minlength=b*h*w*c*c).reshape(b*h*w, c, c).float()
#         #
#     regularisation = alpha*regularisation
#     loss = main_loss + regularisation
#     #
#     return loss, main_loss, regularisation


def noisy_label_loss_v8(pred, preds_noisy, labels):
    #
    main_loss = 0.0
    regularisation = 0.0
    # gradient_coefficient = 0.0
    # for multi-class:
    pred = nn.Softmax(dim=1)(pred)
    b, c, h, w = pred.size()
    #
    for ps_, ts_ in zip(preds_noisy, labels):
        #
        ps_ = nn.Softmax(dim=1)(ps_)
        # current_foreground = 0.0
        #
        class_index = 0
        # for class_index in range(c):
            #
            # ts_temp = torch.zeros_like(ts_).to(device='cuda', dtype=torch.float32)
            # ts_temp[ts_ == current_foreground] = 1.0
            # ts_temp_negative = torch.ones_like(ts_temp) - ts_temp
            #
        ts_temp = ts_
        ts_temp_negative = torch.ones_like(ts_temp) - ts_temp
        #
        ts_positive_ratio = ts_temp.sum() / (b * h * w)
        ts_negative_ratio = (b * h * w - ts_temp.sum()) / (b * h * w)
        #
        p_positive = pred[:, class_index, :, :].clone().reshape(b, 1, h, w)
        p_negative = torch.ones_like(p_positive) - p_positive
        #
        ps_positive = ps_[:, class_index, :, :].clone().reshape(b, 1, h, w)
        ps_negative = torch.ones_like(ps_positive) - ps_positive
        #
        cm00 = p_positive * ps_positive * ts_positive_ratio
        cm01 = p_positive * ps_negative * ts_positive_ratio
        cm10 = p_negative * ps_positive * ts_negative_ratio
        cm11 = p_negative * ps_negative * ts_negative_ratio
        #
        ps_new_fp = cm00 * ps_positive + cm01 * ps_negative
        loss_fp = dice_loss(ps_new_fp, ts_temp)
        #
        ps_new_fn = cm10 * ps_positive + cm11 * ps_negative
        loss_fn = dice_loss(ps_new_fn, ts_temp)
        loss_ps = dice_loss(ps_positive, ts_temp)
        #
        main_loss += loss_fp
        main_loss += loss_fn
        #
        gradient_1 = torch.autograd.grad(outputs=loss_fp, inputs=ps_new_fp,
                                         create_graph=True,
                                         only_inputs=True)[0]
        #
        gradient_2 = torch.autograd.grad(outputs=loss_fn, inputs=ps_new_fn,
                                         create_graph=True,
                                         only_inputs=True)[0]
        #
        gradient_3 = torch.autograd.grad(outputs=loss_ps, inputs=ps_positive,
                                         create_graph=True,
                                         only_inputs=True)[0]
        #
        # regularisation += ((gradient_1.norm(2, dim=1) - 1) ** 2).mean()
        # regularisation += ((gradient_2.norm(2, dim=1) - 1) ** 2).mean()
        regularisation += loss_ps*((gradient_1.norm(2, dim=1) + gradient_2.norm(2, dim=1) + 1e-8) / (gradient_3.norm(2, dim=1) + 1e-8)).mean()
        # regularisation += ((((gradient_1.norm(2, dim=1) + gradient_2.norm(2, dim=1)) / gradient_3.norm(2, dim=1)) - 1.0)**2).mean()
        #
    alpha = 0.01
    regularisation = alpha * regularisation / len(preds_noisy)
    main_loss = main_loss / (len(preds_noisy)*c)
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def noisy_label_loss_v10(pred, preds_noisy, labels, alpha=0):
    #
    main_loss = 0.0
    regularisation = 0.0
    pred = nn.Softmax(dim=1)(pred)
    b, c, h, w = pred.size()
    #
    for ps_, ts_ in zip(preds_noisy, labels):
        #
        ps_ = nn.Softmax(dim=1)(ps_)
        #
        current_foreground = 0.0
        #
        for class_index in range(c - 1):
            #
            ts_temp = torch.zeros_like(ts_).to(device='cuda', dtype=torch.float32)
            ts_temp[ts_ == current_foreground] = 1.0
            ts_temp_negative = torch.ones_like(ts_temp) - ts_temp
            #
            ts_positive_ratio = ts_temp.sum() / (b * h * w)
            ts_negative_ratio = (b * h * w - ts_temp.sum()) / (b * h * w)
            #
            p_positive = pred[:, class_index, :, :].clone().reshape(b, 1, h, w)
            p_negative = torch.ones_like(p_positive) - p_positive
            #
            ps_positive = ps_[:, class_index, :, :].clone().reshape(b, 1, h, w)
            ps_negative = torch.ones_like(ps_positive) - ps_positive
            #
            cm00 = p_positive * ps_positive * ts_positive_ratio
            cm01 = p_positive * ps_negative * ts_positive_ratio
            cm10 = p_negative * ps_positive * ts_negative_ratio
            cm11 = p_negative * ps_negative * ts_negative_ratio
            #
            ps_new_fp = cm00 * ps_positive + cm01 * ps_negative
            loss_fp = dice_loss(ps_new_fp, ts_temp)
            #
            ps_new_fn = cm10 * ps_positive + cm11 * ps_negative
            loss_fn = dice_loss(ps_new_fn, ts_temp)
            #
            loss_ps = dice_loss(ps_positive, ts_temp)
            #
            main_loss += loss_fp
            main_loss += loss_fn
            # main_loss += loss_ps
            #
            gradient_1 = torch.autograd.grad(outputs=loss_fp, inputs=ps_new_fp,
                                             create_graph=True,
                                             only_inputs=True)[0]
            #
            gradient_2 = torch.autograd.grad(outputs=loss_fn, inputs=ps_new_fn,
                                             create_graph=True,
                                             only_inputs=True)[0]
            #
            gradient_3 = torch.autograd.grad(outputs=loss_ps, inputs=ps_positive,
                                             create_graph=True,
                                             only_inputs=True)[0]
            #
            # regularisation += ((gradient_1.norm(2, dim=1) - 1) ** 2).mean()
            # regularisation += ((gradient_2.norm(2, dim=1) - 1) ** 2).mean()
            # regularisation += loss_ps*((gradient_1.norm(2, dim=1) + gradient_2.norm(2, dim=1) + 1e-8) / (gradient_3.norm(2, dim=1) + 1e-8)).mean()
            regularisation += ((((gradient_1.norm(2, dim=1) + gradient_2.norm(2, dim=1)) / gradient_3.norm(2, dim=1)) - 1.0)**2).mean()
            current_foreground += 1.0
            #
    # alpha = 0.01
    regularisation = alpha * regularisation / len(preds_noisy)
    main_loss = main_loss / (len(preds_noisy)*c)
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation