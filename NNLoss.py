import torch
import torch.nn as nn
import torch.nn.functional as F
torch.backends.cudnn.deterministic = True
# ==============================================================================


def dice_loss(input, target):
    smooth = 1
    # input = F.softmax(input, dim=1)
    # input = torch.sigmoid(input) #for binary
    iflat = input.view(-1)
    tflat = target.view(-1)
    intersection = (iflat * tflat).sum()
    union = iflat.sum() + tflat.sum()
    # union = (torch.mul(iflat, iflat) + torch.mul(tflat, tflat)).sum()
    dice_score = (2.*intersection + smooth)/(union + smooth)
    return 1-dice_score


class focal_loss(nn.Module):
    def __init__(self, alpha=1, gamma=2, logits=False, reduce=True):
        super(focal_loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduce=False)
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduce=False)
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1-pt)**self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


def kt_loss(student_output_digits, teacher_output_digits, tempearture):
    # The KL Divergence for PyTorch comparing the softmaxs of teacher and student expects the input tensor to be log probabilities!
    knowledge_transfer_loss = nn.KLDivLoss()(F.logsigmoid(student_output_digits / tempearture), torch.sigmoid(teacher_output_digits / tempearture)) * (tempearture * tempearture)
    return knowledge_transfer_loss


def noisy_label_loss(pred, cm1, cm2, cm3, cm4, target_1, target_2, target_3, target_4):
    #
    b, c, h, w = pred.size()
    #
    confusion_1 = torch.empty(2, 2).to(device='cuda', dtype=torch.float32)
    confusion_2 = torch.empty(2, 2).to(device='cuda', dtype=torch.float32)
    confusion_3 = torch.empty(2, 2).to(device='cuda', dtype=torch.float32)
    confusion_4 = torch.empty(2, 2).to(device='cuda', dtype=torch.float32)
    #
    pred_negative = torch.ones_like(pred) - pred
    #
    target_1_negative = torch.ones_like(target_1) - target_1
    #
    target_2_negative = torch.ones_like(target_2) - target_2
    #
    target_3_negative = torch.ones_like(target_3) - target_3
    #
    target_4_negative = torch.ones_like(target_4) - target_4
    #
    # confusion matrices:
    positive_samples = target_1.sum()
    negative_samples = b*c*h*w - positive_samples
    tp_1 = (pred * target_1).sum() / positive_samples
    fn_1 = (target_1 * pred_negative).sum() / positive_samples
    fp_1 = (pred * target_1_negative).sum() / negative_samples
    tn_1 = (pred_negative * target_1_negative).sum() / negative_samples
    #
    positive_samples = target_2.sum()
    negative_samples = b*c*h*w - positive_samples
    tp_2 = (pred * target_2).sum() / positive_samples
    fn_2 = (target_2 * pred_negative).sum() / positive_samples
    fp_2 = (pred * target_2_negative).sum() / negative_samples
    tn_2 = (pred_negative * target_2_negative).sum() / negative_samples
    #
    positive_samples = target_3.sum()
    negative_samples = b*c*h*w - positive_samples
    tp_3 = (pred * target_3).sum() / positive_samples
    fn_3 = (target_3 * pred_negative).sum() / positive_samples
    fp_3 = (pred * target_3_negative).sum() / negative_samples
    tn_3 = (pred_negative * target_3_negative).sum() / negative_samples
    #
    positive_samples = target_4.sum()
    negative_samples = b*c*h*w - positive_samples
    tp_4 = (pred * target_4).sum() / positive_samples
    fn_4 = (target_4 * pred_negative).sum() / positive_samples
    fp_4 = (pred * target_4_negative).sum() / negative_samples
    tn_4 = (pred_negative * target_4_negative).sum() / negative_samples
    #
    confusion_1[0, 0] = tp_1.clone()
    confusion_1[0, 1] = fn_1.clone()
    confusion_1[1, 0] = fp_1.clone()
    confusion_1[1, 1] = tn_1.clone()
    #
    confusion_2[0, 0] = tp_2.clone()
    confusion_2[0, 1] = fn_2.clone()
    confusion_2[1, 0] = fp_2.clone()
    confusion_2[1, 1] = tn_2.clone()
    #
    confusion_3[0, 0] = tp_3.clone()
    confusion_3[0, 1] = fn_3.clone()
    confusion_3[1, 0] = fp_3.clone()
    confusion_3[1, 1] = tn_3.clone()
    #
    confusion_4[0, 0] = tp_4.clone()
    confusion_4[0, 1] = fn_4.clone()
    confusion_4[1, 0] = fp_4.clone()
    confusion_4[1, 1] = tn_4.clone()
    #
    # print(pred.size())
    #
    pred = pred.reshape(b, h, w, c)
    pred_negative = pred_negative.reshape(b, h, w, c)
    #
    pred = torch.cat([pred, pred_negative], dim=3)
    #
    pred_1 = torch.matmul(pred, confusion_1)
    #
    pred_2 = torch.matmul(pred, confusion_2)
    #
    pred_3 = torch.matmul(pred, confusion_3)
    #
    pred_4 = torch.matmul(pred, confusion_4)
    #
    pred_1 = pred_1[:, :, :, 0].clone().reshape(b, c, h, w)
    #
    pred_2 = pred_2[:, :, :, 0].clone().reshape(b, c, h, w)
    #
    pred_3 = pred_3[:, :, :, 0].clone().reshape(b, c, h, w)
    #
    pred_4 = pred_4[:, :, :, 0].clone().reshape(b, c, h, w)
    #
    # ce_1 = nn.BCELoss(reduction='mean')(pred_1, target_1)
    # ce_2 = nn.BCELoss(reduction='mean')(pred_2, target_2)
    # ce_3 = nn.BCELoss(reduction='mean')(pred_3, target_3)
    # ce_4 = nn.BCELoss(reduction='mean')(pred_4, target_4)
    #
    ce_1 = dice_loss(pred_1, target_1)
    ce_2 = dice_loss(pred_2, target_2)
    ce_3 = dice_loss(pred_3, target_3)
    ce_4 = dice_loss(pred_4, target_4)
    #
    # ce = ce_1 + ce_2 + ce_3 + ce_4
    ce = (ce_1 + ce_2 + ce_3 + ce_4) / 4
    #
    # trace = tp_1 + tp_2 + tp_3 + tn_1 + tn_2 + tn_3 + tp_4 + tn_4
    #
    positive_frequency = positive_samples / (b*c*h*w)
    negative_frequency = negative_samples / (b*c*h*w)
    #
    trace = (tp_1 + tp_2 + tp_3 + tp_4) / positive_frequency + (tn_1 + tn_2 + tn_3 + tn_4) / negative_frequency
    #
    trace = 1 / trace
    #
    alpha = 1
    #
    y = ce + alpha * trace
    #
    return y, ce, alpha * trace


def noisy_label_loss_v2(pred, target_1, target_2, target_3, target_4):
    #
    b, c, h, w = pred.size()
    #
    pred_1 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_2 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_3 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_4 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    #
    trace = 0
    #
    confusion_1 = torch.empty(b, 2, 2).to(device='cuda', dtype=torch.float32)
    confusion_2 = torch.empty(b, 2, 2).to(device='cuda', dtype=torch.float32)
    confusion_3 = torch.empty(b, 2, 2).to(device='cuda', dtype=torch.float32)
    confusion_4 = torch.empty(b, 2, 2).to(device='cuda', dtype=torch.float32)
    #
    for batch in range(0, b):
        #
        pred_slice = pred[batch, :, :, :].clone().squeeze(0).squeeze(1)
        #
        assert h, w == pred_slice.size()
        #
        pred_slice_negative = torch.ones_like(pred_slice) - pred_slice
        #
        target_1_slice = target_1[batch, :, :, :].clone().squeeze(0).squeeze(1)
        #
        target_1_slice_negative = torch.ones_like(target_1_slice) - target_1_slice
        #
        target_2_slice = target_2[batch, :, :, :].clone().squeeze(0).squeeze(1)
        #
        target_2_slice_negative = torch.ones_like(target_2_slice) - target_2_slice
        #
        target_3_slice = target_3[batch, :, :, :].clone().squeeze(0).squeeze(1)
        #
        target_3_slice_negative = torch.ones_like(target_3_slice) - target_3_slice
        #
        target_4_slice = target_4[batch, :, :, :].clone().squeeze(0).squeeze(1)
        #
        target_4_slice_negative = torch.ones_like(target_4_slice) - target_4_slice
        #
        # confusion matrices:
        positive_samples = target_1_slice.sum()
        negative_samples = c*h*w - positive_samples
        tp_1 = (pred_slice * target_1_slice).sum() / positive_samples
        fn_1 = (target_1_slice * pred_slice_negative).sum() / positive_samples
        fp_1 = (pred_slice * target_1_slice_negative).sum() / negative_samples
        tn_1 = (pred_slice_negative * target_1_slice_negative).sum() / negative_samples
        #
        trace += tp_1
        trace += tn_1
        #
        positive_samples = target_2_slice.sum()
        negative_samples = c*h*w - positive_samples
        #
        tp_2 = (pred_slice * target_2_slice).sum() / positive_samples
        fn_2 = (target_2_slice * pred_slice_negative).sum() / positive_samples
        fp_2 = (pred_slice * target_2_slice_negative).sum() / negative_samples
        tn_2 = (pred_slice_negative * target_2_slice_negative).sum() / negative_samples
        #
        trace += tp_2
        trace += tn_2
        #
        positive_samples = target_3_slice.sum()
        negative_samples = c*h*w - positive_samples
        #
        tp_3 = (pred_slice * target_3_slice).sum() / positive_samples
        fn_3 = (target_3_slice * pred_slice_negative).sum() / positive_samples
        fp_3 = (pred_slice * target_3_slice_negative).sum() / negative_samples
        tn_3 = (pred_slice_negative * target_3_slice_negative).sum() / negative_samples
        #
        trace += tp_3
        trace += tn_3
        #
        positive_samples = target_4_slice.sum()
        negative_samples = c*h*w - positive_samples
        #
        tp_4 = (pred_slice * target_4_slice).sum() / positive_samples
        fn_4 = (target_4_slice * pred_slice_negative).sum() / positive_samples
        fp_4 = (pred_slice * target_4_slice_negative).sum() / negative_samples
        tn_4 = (pred_slice_negative * target_4_slice_negative).sum() / negative_samples
        #
        trace += tp_4
        trace += tn_4
        #
        confusion_1[batch, 0, 0] = tp_1.clone()
        confusion_1[batch, 0, 1] = fn_1.clone()
        confusion_1[batch, 1, 0] = fp_1.clone()
        confusion_1[batch, 1, 1] = tn_1.clone()
        #
        confusion_2[batch, 0, 0] = tp_2.clone()
        confusion_2[batch, 0, 1] = fn_2.clone()
        confusion_2[batch, 1, 0] = fp_2.clone()
        confusion_2[batch, 1, 1] = tn_2.clone()
        #
        confusion_3[batch, 0, 0] = tp_3.clone()
        confusion_3[batch, 0, 1] = fn_3.clone()
        confusion_3[batch, 1, 0] = fp_3.clone()
        confusion_3[batch, 1, 1] = tn_3.clone()
        #
        confusion_4[batch, 0, 0] = tp_4.clone()
        confusion_4[batch, 0, 1] = fn_4.clone()
        confusion_4[batch, 1, 0] = fp_4.clone()
        confusion_4[batch, 1, 1] = tn_4.clone()
        #
        pred_slice = pred_slice.reshape(c, h, w, 1)
        #
        pred_slice_negative = pred_slice_negative.reshape(c, h, w, 1)
        #
        pred__ = torch.cat([pred_slice, pred_slice_negative], dim=3)
        #
        pred_1_slice = torch.matmul(pred__, confusion_1[batch, :, :].clone())
        #
        pred_2_slice = torch.matmul(pred__, confusion_2[batch, :, :].clone())
        #
        pred_3_slice = torch.matmul(pred__, confusion_3[batch, :, :].clone())
        #
        pred_4_slice = torch.matmul(pred__, confusion_4[batch, :, :].clone())
        #
        pred_1[batch, :, :, :] = pred_1_slice[:, :, :, 0].clone().reshape(c, h, w)
        #
        pred_2[batch, :, :, :] = pred_2_slice[:, :, :, 0].clone().reshape(c, h, w)
        #
        pred_3[batch, :, :, :] = pred_3_slice[:, :, :, 0].clone().reshape(c, h, w)
        #
        pred_4[batch, :, :, :] = pred_4_slice[:, :, :, 0].clone().reshape(c, h, w)
    #
    # ce_1 = nn.BCELoss(reduction='mean')(pred_1, target_1)
    # ce_2 = nn.BCELoss(reduction='mean')(pred_2, target_2)
    # ce_3 = nn.BCELoss(reduction='mean')(pred_3, target_3)
    # ce_4 = nn.BCELoss(reduction='mean')(pred_4, target_4)
    #
    ce_1 = dice_loss(pred_1, target_1)
    ce_2 = dice_loss(pred_2, target_2)
    ce_3 = dice_loss(pred_3, target_3)
    ce_4 = dice_loss(pred_4, target_4)
    #
    ce = (ce_1 + ce_2 + ce_3 + ce_4) / 4
    #
    trace = 1 / trace
    #
    # if epoch < 200:
    #     #
    #     alpha = 0.00
    #     #
    # else:
    #     alpha = (epoch - 200)*0.0001
    #
    alpha = 0
    #
    y = ce + alpha * trace
    #
    return y, ce, alpha * trace


def noisy_label_loss_v3(pred, cms, target_1, target_2, target_3, target_4):
    #
    b, c, h, w = pred.size()
    #
    pred_1 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_2 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_3 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_4 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    #
    trace = 0
    #
    for batch in range(0, b):
        #
        pred_slice = pred[batch, :, :, :].clone().squeeze(0).squeeze(1)
        pred_slice_negative = torch.ones_like(pred_slice) - pred_slice
        assert h, w == pred_slice.size()
        #
        # confusion matrices:
        cm1 = cms[0][batch, :, :, :].clone().reshape(2, 2)
        cm2 = cms[1][batch, :, :, :].clone().reshape(2, 2)
        cm3 = cms[2][batch, :, :, :].clone().reshape(2, 2)
        cm4 = cms[3][batch, :, :, :].clone().reshape(2, 2)
        #
        cm1[0, 0] = cm1[0, 0].clone() / (cm1[0, 0].clone() + cm1[0, 1].clone())
        cm1[0, 1] = cm1[0, 1].clone() / (cm1[0, 0].clone() + cm1[0, 1].clone())
        cm1[1, 0] = cm1[1, 0].clone() / (cm1[1, 0].clone() + cm1[1, 1].clone())
        cm1[1, 1] = cm1[1, 1].clone() / (cm1[1, 0].clone() + cm1[1, 1].clone())
        #
        cm2[0, 0] = cm2[0, 0].clone() / (cm2[0, 0].clone() + cm2[0, 1].clone())
        cm2[0, 1] = cm2[0, 1].clone() / (cm2[0, 0].clone() + cm2[0, 1].clone())
        cm2[1, 0] = cm2[1, 0].clone() / (cm2[1, 0].clone() + cm2[1, 1].clone())
        cm2[1, 1] = cm2[1, 1].clone() / (cm2[1, 0].clone() + cm2[1, 1].clone())
        #
        cm3[0, 0] = cm3[0, 0].clone() / (cm3[0, 0].clone() + cm3[0, 1].clone())
        cm3[0, 1] = cm3[0, 1].clone() / (cm3[0, 0].clone() + cm3[0, 1].clone())
        cm3[1, 0] = cm3[1, 0].clone() / (cm3[1, 0].clone() + cm3[1, 1].clone())
        cm3[1, 1] = cm3[1, 1].clone() / (cm3[1, 0].clone() + cm3[1, 1].clone())
        #
        cm4[0, 0] = cm4[0, 0].clone() / (cm4[0, 0].clone() + cm4[0, 1].clone())
        cm4[0, 1] = cm4[0, 1].clone() / (cm4[0, 0].clone() + cm4[0, 1].clone())
        cm4[1, 0] = cm4[1, 0].clone() / (cm4[1, 0].clone() + cm4[1, 1].clone())
        cm4[1, 1] = cm4[1, 1].clone() / (cm4[1, 0].clone() + cm4[1, 1].clone())
        #
        trace = cm1[0, 0] + cm1[1, 1] + cm2[0, 0] + cm2[1, 1] + cm3[0, 0] + cm3[1, 1] + cm4[0, 0] + cm4[1, 1]
        #
        pred_slice = pred_slice.reshape(h, w, 1)
        pred_slice_negative = pred_slice_negative.reshape(h, w, 1)
        pred__ = torch.cat([pred_slice, pred_slice_negative], dim=2)
        #
        pred_1_slice = torch.matmul(pred__, cm1)
        pred_2_slice = torch.matmul(pred__, cm2)
        pred_3_slice = torch.matmul(pred__, cm3)
        pred_4_slice = torch.matmul(pred__, cm4)
        #
        pred_1[batch, :, :, :] = pred_1_slice[:, :, 0].clone().reshape(c, h, w)
        pred_2[batch, :, :, :] = pred_2_slice[:, :, 0].clone().reshape(c, h, w)
        pred_3[batch, :, :, :] = pred_3_slice[:, :, 0].clone().reshape(c, h, w)
        pred_4[batch, :, :, :] = pred_4_slice[:, :, 0].clone().reshape(c, h, w)
    #
    # ce_1 = nn.BCELoss(reduction='mean')(pred_1, target_1)
    # ce_2 = nn.BCELoss(reduction='mean')(pred_2, target_2)
    # ce_3 = nn.BCELoss(reduction='mean')(pred_3, target_3)
    # ce_4 = nn.BCELoss(reduction='mean')(pred_4, target_4)
    #
    ce_1 = dice_loss(torch.sigmoid(pred_1), target_1)
    ce_2 = dice_loss(torch.sigmoid(pred_2), target_2)
    ce_3 = dice_loss(torch.sigmoid(pred_3), target_3)
    ce_4 = dice_loss(torch.sigmoid(pred_4), target_4)
    #
    # ce_1 = dice_loss(pred_1, target_1)
    # ce_2 = dice_loss(pred_2, target_2)
    # ce_3 = dice_loss(pred_3, target_3)
    # ce_4 = dice_loss(pred_4, target_4)
    #
    ce = (ce_1 + ce_2 + ce_3 + ce_4) / 4
    #
    trace = 1 / trace
    #
    # if epoch < 200:
    #     #
    #     alpha = 0.00
    #     #
    # else:
    #     alpha = (epoch - 200)*0.0001
    #
    alpha = 0.00
    #
    y = ce + alpha * trace
    #
    return y, ce, alpha * trace


def noisy_label_loss_v4(pred, cms, target_1, target_2, target_3, target_4):
    #
    # output confusion matrices: h x w x c x c
    #
    b, c, h, w = pred.size()
    #
    pred_1 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_2 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_3 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    pred_4 = torch.empty(b, c, h, w).to(device='cuda', dtype=torch.float32)
    #
    trace = 0
    #
    for batch in range(b):
        #
        pred_slice = pred[batch, :, :, :].clone().squeeze(0).squeeze(1)
        pred_slice_negative = torch.ones_like(pred_slice) - pred_slice
        assert h, w == pred_slice.size()
        #
        # confusion matrices:
        cm1 = cms[0][batch, :, :, :].clone().reshape(h, w, 2, 2)
        cm2 = cms[1][batch, :, :, :].clone().reshape(h, w, 2, 2)
        cm3 = cms[2][batch, :, :, :].clone().reshape(h, w, 2, 2)
        cm4 = cms[3][batch, :, :, :].clone().reshape(h, w, 2, 2)
        #
        cm_1_norm = torch.empty(h, w, 2, 2).to(device='cuda', dtype=torch.float32)
        cm_2_norm = torch.empty(h, w, 2, 2).to(device='cuda', dtype=torch.float32)
        cm_3_norm = torch.empty(h, w, 2, 2).to(device='cuda', dtype=torch.float32)
        cm_4_norm = torch.empty(h, w, 2, 2).to(device='cuda', dtype=torch.float32)
        #
        for i in range(h):
            #
            for j in range(w):
                #
                cm_1_norm[i, j, 0, 0] = cm1[i, j, 0, 0].clone() / (cm1[i, j, 0, 0].clone() + cm1[i, j, 0, 1].clone() + 1e-8)
                cm_1_norm[i, j, 0, 1] = cm1[i, j, 0, 1].clone() / (cm1[i, j, 0, 0].clone() + cm1[i, j, 0, 1].clone() + 1e-8)
                cm_1_norm[i, j, 1, 0] = cm1[i, j, 1, 0].clone() / (cm1[i, j, 1, 0].clone() + cm1[i, j, 1, 1].clone() + 1e-8)
                cm_1_norm[i, j, 1, 1] = cm1[i, j, 1, 1].clone() / (cm1[i, j, 1, 0].clone() + cm1[i, j, 1, 1].clone() + 1e-8)
                #
                cm_2_norm[i, j, 0, 0] = cm2[i, j, 0, 0].clone() / (cm2[i, j, 0, 0].clone() + cm2[i, j, 0, 1].clone() + 1e-8)
                cm_2_norm[i, j, 0, 1] = cm2[i, j, 0, 1].clone() / (cm2[i, j, 0, 0].clone() + cm2[i, j, 0, 1].clone() + 1e-8)
                cm_2_norm[i, j, 1, 0] = cm2[i, j, 1, 0].clone() / (cm2[i, j, 1, 0].clone() + cm2[i, j, 1, 1].clone() + 1e-8)
                cm_2_norm[i, j, 1, 1] = cm2[i, j, 1, 1].clone() / (cm2[i, j, 1, 0].clone() + cm2[i, j, 1, 1].clone() + 1e-8)
                #
                cm_3_norm[i, j, 0, 0] = cm3[i, j, 0, 0].clone() / (cm3[i, j, 0, 0].clone() + cm3[i, j, 0, 1].clone() + 1e-8)
                cm_3_norm[i, j, 0, 1] = cm3[i, j, 0, 1].clone() / (cm3[i, j, 0, 0].clone() + cm3[i, j, 0, 1].clone() + 1e-8)
                cm_3_norm[i, j, 1, 0] = cm3[i, j, 1, 0].clone() / (cm3[i, j, 1, 0].clone() + cm3[i, j, 1, 1].clone() + 1e-8)
                cm_3_norm[i, j, 1, 1] = cm3[i, j, 1, 1].clone() / (cm3[i, j, 1, 0].clone() + cm3[i, j, 1, 1].clone() + 1e-8)
                #
                cm_4_norm[i, j, 0, 0] = cm4[i, j, 0, 0].clone() / (cm4[i, j, 0, 0].clone() + cm4[i, j, 0, 1].clone() + 1e-8)
                cm_4_norm[i, j, 0, 1] = cm4[i, j, 0, 1].clone() / (cm4[i, j, 0, 0].clone() + cm4[i, j, 0, 1].clone() + 1e-8)
                cm_4_norm[i, j, 1, 0] = cm4[i, j, 1, 0].clone() / (cm4[i, j, 1, 0].clone() + cm4[i, j, 1, 1].clone() + 1e-8)
                cm_4_norm[i, j, 1, 1] = cm4[i, j, 1, 1].clone() / (cm4[i, j, 1, 0].clone() + cm4[i, j, 1, 1].clone() + 1e-8)
                #
                trace = cm_1_norm[i, j, 0, 0].clone() + \
                        cm_1_norm[i, j, 1, 1].clone() + \
                        cm_2_norm[i, j, 0, 0].clone() + \
                        cm_2_norm[i, j, 1, 1].clone() + \
                        cm_3_norm[i, j, 0, 0].clone() + \
                        cm_3_norm[i, j, 1, 1].clone() + \
                        cm_4_norm[i, j, 0, 0].clone() + \
                        cm_4_norm[i, j, 1, 1].clone()
                #
                pred__ = torch.empty(1, 2).to(device='cuda', dtype=torch.float32)
                #
                # pred_slice_position = pred_slice[i, j].clone().reshape(1, 1, 1)
                # pred_slice_negative_position = pred_slice_negative[i, j].clone().reshape(1, 1, 1)
                #
                # pred__ = torch.cat([pred_slice_position, pred_slice_negative_position], dim=2)
                pred__[0, 0] = pred_slice[i, j].clone()
                pred__[0, 1] = pred_slice_negative[i, j].clone()
                #
                pred_1_slice_position = torch.matmul(pred__, cm_1_norm[i, j, :, :].clone().reshape(2, 2))
                pred_2_slice_position = torch.matmul(pred__, cm_2_norm[i, j, :, :].clone().reshape(2, 2))
                pred_3_slice_position = torch.matmul(pred__, cm_3_norm[i, j, :, :].clone().reshape(2, 2))
                pred_4_slice_position = torch.matmul(pred__, cm_4_norm[i, j, :, :].clone().reshape(2, 2))
                #
                pred_1[batch, :, i, j] = pred_1_slice_position[0, 0].clone()
                pred_2[batch, :, i, j] = pred_2_slice_position[0, 0].clone()
                pred_3[batch, :, i, j] = pred_3_slice_position[0, 0].clone()
                pred_4[batch, :, i, j] = pred_4_slice_position[0, 0].clone()

    # ce_1 = nn.BCELoss(reduction='mean')(pred_1, target_1)
    # ce_2 = nn.BCELoss(reduction='mean')(pred_2, target_2)
    # ce_3 = nn.BCELoss(reduction='mean')(pred_3, target_3)
    # ce_4 = nn.BCELoss(reduction='mean')(pred_4, target_4)
    #
    ce_1 = dice_loss(torch.sigmoid(pred_1), target_1)
    ce_2 = dice_loss(torch.sigmoid(pred_2), target_2)
    ce_3 = dice_loss(torch.sigmoid(pred_3), target_3)
    ce_4 = dice_loss(torch.sigmoid(pred_4), target_4)
    #
    ce = (ce_1 + ce_2 + ce_3 + ce_4) / 4
    #
    trace = 1 / trace
    #
    # if epoch < 200:
    #     #
    #     alpha = 0.00
    #     #
    # else:
    #     alpha = (epoch - 200)*0.0001
    #
    alpha = 0.00
    #
    y = ce + alpha * trace
    #
    return y, ce, alpha * trace


def noisy_label_loss_v5(pred, cms, target_1, target_2, target_3, target_4):
    #
    pred_1 = torch.sigmoid(pred*cms[0])
    pred_2 = torch.sigmoid(pred*cms[1])
    pred_3 = torch.sigmoid(pred*cms[2])
    pred_4 = torch.sigmoid(pred*cms[3])
    #
    # ce_1 = nn.BCELoss(reduction='mean')(pred_1, target_1)
    # ce_2 = nn.BCELoss(reduction='mean')(pred_2, target_2)
    # ce_3 = nn.BCELoss(reduction='mean')(pred_3, target_3)
    # ce_4 = nn.BCELoss(reduction='mean')(pred_4, target_4)
    #
    ce_1 = dice_loss(pred_1, target_1)
    ce_2 = dice_loss(pred_2, target_2)
    ce_3 = dice_loss(pred_3, target_3)
    ce_4 = dice_loss(pred_4, target_4)
    #
    # ce_1 = dice_loss(torch.sigmoid(pred_1), target_1)
    # ce_2 = dice_loss(torch.sigmoid(pred_2), target_2)
    # ce_3 = dice_loss(torch.sigmoid(pred_3), target_3)
    # ce_4 = dice_loss(torch.sigmoid(pred_4), target_4)
    #
    ce = (ce_1 + ce_2 + ce_3 + ce_4) / 4
    #
    # trace = cms[0].sum() + cms[1].sum() + cms[2].sum() + cms[3].sum()
    #
    trace = 0.00
    #
    alpha = 0.01
    #
    y = ce + alpha * trace
    #
    return y, ce, alpha * trace


def noisy_label_loss_v6(pred, cm1, cm2, cm3, cm4, target_1, target_2, target_3, target_4):
    #
    # cm: confusion matrix 1 at Batch x Channel**2 x Height x Width
    # pred: prediction digits at Batch x Channel x Height x Width
    # target: ground truth at Batch x Channel x Height x Width
    #
    b, c, h, w = pred.size()
    #
    cm1 = cm1.reshape(b * h * w, 2, 2)
    cm2 = cm2.reshape(b * h * w, 2, 2)
    cm3 = cm3.reshape(b * h * w, 2, 2)
    cm4 = cm4.reshape(b * h * w, 2, 2)
    #
    pred_positive = torch.sigmoid(pred)
    pred_negative = torch.ones_like(pred_positive) - pred_positive
    #
    pred_class = torch.cat([pred_positive, pred_negative], dim=1)
    pred_class = pred_class.reshape(b * h * w, 2, 1)
    #
    sum_each_rows_1 = cm1.sum(1, keepdim=True)
    sum_each_rows_2 = cm2.sum(1, keepdim=True)
    sum_each_rows_3 = cm3.sum(1, keepdim=True)
    sum_each_rows_4 = cm4.sum(1, keepdim=True)
    #
    cm1 = cm1 / sum_each_rows_1
    cm2 = cm2 / sum_each_rows_2
    cm3 = cm3 / sum_each_rows_3
    cm4 = cm4 / sum_each_rows_4
    #
    # cm1 = F.softmax(cm1, dim=1)
    # cm2 = F.softmax(cm2, dim=1)
    # cm3 = F.softmax(cm3, dim=1)
    # cm4 = F.softmax(cm4, dim=1)
    #
    print('Confusion matrices:')
    print(cm1[0, :, :])
    print(cm2[0, :, :])
    print(cm3[0, :, :])
    print(cm4[0, :, :])
    #
    trace = cm1[:, 0, 0].mean() + cm1[:, 1, 1].mean() + cm2[:, 0, 0].mean() + cm2[:, 1, 1].mean() + cm3[:, 0, 0].mean() + cm3[:, 1, 1].mean() + cm4[:, 0, 0].mean() + cm4[:, 1, 1].mean()
    trace = 1 / trace
    #
    pred_1 = torch.bmm(cm1, pred_class).reshape(b, 2, h, w)[:, 0, :, :].clone().reshape(b, 1, h, w)
    pred_2 = torch.bmm(cm2, pred_class).reshape(b, 2, h, w)[:, 0, :, :].clone().reshape(b, 1, h, w)
    pred_3 = torch.bmm(cm3, pred_class).reshape(b, 2, h, w)[:, 0, :, :].clone().reshape(b, 1, h, w)
    pred_4 = torch.bmm(cm4, pred_class).reshape(b, 2, h, w)[:, 0, :, :].clone().reshape(b, 1, h, w)
    #
    #
    # print(pred_1.size())
    # print(target_1.size())
    #
    ce_1 = dice_loss(pred_1, target_1)
    ce_2 = dice_loss(pred_2, target_2)
    ce_3 = dice_loss(pred_3, target_3)
    ce_4 = dice_loss(pred_4, target_4)
    #
    ce = (ce_1 + ce_2 + ce_3 + ce_4) / 4
    #
    # trace = 0.00
    #
    alpha = 0.1
    #
    y = ce + alpha * trace
    #
    return y, ce, alpha * trace


def noisy_label_loss_v7(p, p1, p2, p3, p4, target_1, target_2, target_3, target_4):
    #
    b, c, h, w = p.size()
    #
    p = torch.sigmoid(p)
    p_negative = torch.ones_like(p) - p
    pp = torch.cat([p, p_negative], dim=1)
    #
    p1 = torch.sigmoid(p1)
    p2 = torch.sigmoid(p2)
    p3 = torch.sigmoid(p3)
    p4 = torch.sigmoid(p4)
    #
    p1_ = torch.ones_like(p1) - p1
    p2_ = torch.ones_like(p2) - p2
    p3_ = torch.ones_like(p3) - p3
    p4_ = torch.ones_like(p4) - p4
    #
    p1_ = torch.cat([p1, p1_], dim=1)
    p2_ = torch.cat([p2, p2_], dim=1)
    p3_ = torch.cat([p3, p3_], dim=1)
    p4_ = torch.cat([p4, p4_], dim=1)
    #
    p_threshold = (p > 0.5).float()
    p1 = (p1 > 0.5).float()
    p2 = (p2 > 0.5).float()
    p3 = (p3 > 0.5).float()
    p4 = (p4 > 0.5).float()
    #
    N = torch.tensor(2, dtype=torch.float32)
    #
    indices = N * (p_threshold + 1.0).view(-1) + (p1 + 1.0).view(-1)
    cm1 = torch.bincount(indices.long(), minlength=b*h*w*2*2).reshape(b*h*w, 2, 2)
    indices = N * p_threshold.view(-1) + p2.view(-1)
    cm2 = torch.bincount(indices.long(), minlength=b*h*w*2*2).reshape(b*h*w, 2, 2)
    indices = N * p_threshold.view(-1) + p3.view(-1)
    cm3 = torch.bincount(indices.long(), minlength=b*h*w*2*2).reshape(b*h*w, 2, 2)
    indices = N * p_threshold.view(-1) + p4.view(-1)
    cm4 = torch.bincount(indices.long(), minlength=b*h*w*2*2).reshape(b*h*w, 2, 2)
    #
    p1_p = torch.bmm(cm1.float(), pp.reshape(b*h*w, 2, 1)).reshape(b, 2, h, w)[:, 0, :, :].clone().reshape(b, 1, h, w)
    #
    # ce = dice_loss(p1_p, target_1)
    ce = nn.BCEWithLogitsLoss(reduction='mean')(p1_p, target_1)
    # print(ce)
    #
    trace = 0
    alpha = 0
    #
    return p, ce, alpha * trace