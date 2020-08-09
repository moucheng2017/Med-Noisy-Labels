import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================


def noisy_label_loss(pred, cms, labels, alpha=0.1):
    # Proposed novel loss function for learning noisy labels in segmentation task
    '''
    :param pred: predicted ground truth
    :param cms: confusion matrices
    :param labels: noisy labels
    :param alpha: weight for regularisation
    :return:
    '''
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()
    pred_norm = nn.Softmax(dim=1)(pred)
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)
    #
    for cm, label_noisy in zip(cms, labels):
        #
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)
        #
        cm = cm / cm.sum(1, keepdim=True)
        #
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        #
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
        #
    regularisation = alpha*regularisation
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


# def noisy_label_loss_global(pred, cms, labels, alpha=0.1):
#     # Proposed novel loss function for learning noisy labels in segmentation task
#     '''
#     :param pred: predicted ground truth
#     :param cms: confusion matrices: b x c x c
#     :param labels: noisy labels
#     :param alpha: weight for regularisation
#     :return:
#     '''
#     main_loss = 0.0
#     regularisation = 0.0
#     b, c, h, w = pred.size()
#     pred_norm = nn.Softmax(dim=1)(pred)
#     pred_norm = pred_norm.view(b, c, h*w) # b x c x h*w
#     #
#     for cm, label_noisy in zip(cms, labels):
#         # cm: b x c x c
#         cm = cm / cm.sum(1, keepdim=True)
#         pred_noisy = torch.bmm(cm, pred_norm) # b x c x h*w
#         pred_noisy = pred_noisy.view(b, c, h, w)
#         loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
#         #
#         main_loss += loss_current
#         regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)
#         #
#     regularisation = alpha*regularisation
#     loss = main_loss + regularisation
#     #
#     return loss, main_loss, regularisation


def noisy_label_loss_low_rank(pred, cms, labels, alpha):
    # pred: prediction for true segmentation
    # cms: confusion matrices for each annotators
    # alpha: weight for trace
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
        #
        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2
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
        regularisation_ = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)

        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())

        regularisation += regularisation_

        main_loss += loss_current
        #
    regularisation = alpha*regularisation
    #
    loss = main_loss + regularisation
    #
    return loss, main_loss, regularisation


def dice_loss(input, target):
    smooth = 1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


