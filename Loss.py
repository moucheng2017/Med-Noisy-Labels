import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# =======================================


def noisy_label_loss(pred, cms, labels, alpha=0.1):
    """ This function defines the proposed trace regularised loss function, suitable for either binary
    or multi-class segmentation task. Essentially, each pixel has a confusion matrix.

    Args:
        pred (torch.tensor): output tensor of the last layer of the segmentation network without Sigmoid or Softmax
        cms (list): a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        labels (torch.tensor): labels
        alpha (double): a hyper-parameter to decide the strength of regularisation

    Returns:
        loss (double): total loss value, sum between main_loss and regularisation
        main_loss (double): main segmentation loss
        regularisation (double): regularisation loss

    """
    main_loss = 0.0
    regularisation = 0.0
    b, c, h, w = pred.size()

    # normalise the segmentation output tensor along dimension 1
    pred_norm = nn.Softmax(dim=1)(pred)

    # b x c x h x w ---> b*h*w x c x 1
    pred_norm = pred_norm.view(b, c, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c, 1)

    for cm, label_noisy in zip(cms, labels):
        # cm: learnt confusion matrix for each noisy label, b x c**2 x h x w
        # label_noisy: noisy label, b x h x w

        # b x c**2 x h x w ---> b*h*w x c x c
        cm = cm.view(b, c ** 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, c * c).view(b * h * w, c, c)

        # normalisation along the rows:
        cm = cm / cm.sum(1, keepdim=True)

        # matrix multiplication to calculate the predicted noisy segmentation:
        # cm: b*h*w x c x c
        # pred_noisy: b*h*w x c x 1
        pred_noisy = torch.bmm(cm, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())
        main_loss += loss_current
        regularisation += torch.trace(torch.transpose(torch.sum(cm, dim=0), 0, 1)).sum() / (b * h * w)

    regularisation = alpha*regularisation
    loss = main_loss + regularisation

    return loss, main_loss, regularisation


def noisy_label_loss_low_rank(pred, cms, labels, alpha):
    """ This function defines the proposed low-rank trace regularised loss function, suitable for either binary
    or multi-class segmentation task. Essentially, each pixel has a confusion matrix.

    Args:
        pred (torch.tensor): output tensor of the last layer of the segmentation network without Sigmoid or Softmax
        cms (list): a list of output tensors for each noisy label, each item contains all of the modelled confusion matrix for each spatial location
        labels (torch.tensor): labels
        alpha (double): a hyper-parameter to decide the strength of regularisation

    Returns:
        loss (double): total loss value, sum between main_loss and regularisation
        main_loss (double): main segmentation loss
        regularisation (double): regularisation loss

    """

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
        # cm: learnt confusion matrix for each noisy label, b x c_r_d x h x w, where c_r_d < c
        # label_noisy: noisy label, b x h x w

        b, c_r_d, h, w = cm.size()
        r = c_r_d // c // 2

        # reconstruct the full-rank confusion matrix from low-rank approximations:
        cm1 = cm[:, 0:r * c, :, :]
        cm2 = cm[:, r * c:c_r_d-1, :, :]
        scaling_factor = cm[:, c_r_d-1, :, :].view(b, 1, h, w).view(b, 1, h*w).permute(0, 2, 1).contiguous().view(b*h*w, 1, 1)
        cm1_reshape = cm1.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, r, c)
        cm2_reshape = cm2.view(b, c_r_d // 2, h * w).permute(0, 2, 1).contiguous().view(b * h * w, r * c).view(b * h * w, c, r)
        cm_reconstruct = torch.bmm(cm2_reshape, cm1_reshape)

        # add an identity residual to make approximation easier
        identity_residual = torch.cat(b*h*w*[torch.eye(c, c)]).reshape(b*h*w, c, c).to(device='cuda', dtype=torch.float32)
        cm_reconstruct_approx = cm_reconstruct + identity_residual*scaling_factor
        cm_reconstruct_approx = cm_reconstruct_approx / cm_reconstruct_approx.sum(1, keepdim=True)

        # calculate noisy prediction from confusion matrix and prediction
        pred_noisy = torch.bmm(cm_reconstruct_approx, pred_norm).view(b*h*w, c)
        pred_noisy = pred_noisy.view(b, h * w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)

        regularisation_ = torch.trace(torch.transpose(torch.sum(cm_reconstruct_approx, dim=0), 0, 1)).sum() / (b * h * w)

        loss_current = nn.CrossEntropyLoss(reduction='mean')(pred_noisy, label_noisy.view(b, h, w).long())

        regularisation += regularisation_

        main_loss += loss_current

    regularisation = alpha*regularisation

    loss = main_loss + regularisation

    return loss, main_loss, regularisation


def dice_loss(input, target):
    """ This is a normal dice loss function for binary segmentation.

    Args:
        input: output of the segmentation network
        target: ground truth label

    Returns:
        dice score

    """
    smooth = 1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


