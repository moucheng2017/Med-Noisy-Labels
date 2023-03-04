# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.14.5
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

import os
import errno
import torch
import timeit
import imageio
import numpy as np
import torch.nn as nn
from adamW import AdamW
from torch.utils import data
import matplotlib.pyplot as plt
from tensorboardX import SummaryWriter
from Loss import noisy_label_loss
from Utilis import segmentation_scores, CustomDataset_punet, calculate_cm
from Utilis import evaluate_noisy_label_4, evaluate_noisy_label_5, evaluate_noisy_label_6
# our proposed model:
from Models import UNet_CMs, UNet_GlobalCMs
from PIL import Image

# +
# ========================= #
# Hyper-parameters setting
# ========================= #

# hyper-parameters for model:
input_dim = 3 # dimension of input
width = 24 # width of the network
depth = 3 # depth of the network, downsampling times is (depth-1)
class_no = 2 # class number, 2 for binary

# hyper-parameters for training:
train_batchsize = 5 # batch size
alpha = 0.001 # weight of the trace regularisation of learnt confusion matrices
num_epochs = 20 # total epochs
learning_rate = 1e-2 # learning rate

# +
# ======================================= #
# Prepare a few data examples from MNIST 
# ======================================= #

# Change path for your own datasets here:
#data_path = './MNIST_examples'
#dataset_tag = 'mnist'
#label_mode = 'multi'

### INSERT CODE ###
data_path = './oocytes_gent/'
dataset_tag = 'oocytes_gent'
label_mode = 'multi'
### ----------- ###

# full path to train/validate/test:
test_path = data_path + '/test' 
train_path = data_path + '/train'
validate_path = data_path + '/validate'

# prepare data sets using our customdataset
train_dataset = CustomDataset_punet(dataset_location=train_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=True)
validate_dataset = CustomDataset_punet(dataset_location=validate_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)
test_dataset = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel=label_mode, augmentation=False)

# putting dataset into data loaders
trainloader = data.DataLoader(train_dataset, batch_size=train_batchsize, shuffle=True, num_workers=2, drop_last=True)
validateloader = data.DataLoader(validate_dataset, batch_size=1, shuffle=False, drop_last=False)
testloader = data.DataLoader(test_dataset, batch_size=1, shuffle=False, drop_last=False)

# demonstrate the training samples:
Image_index_to_demonstrate = 3
images, labels_AR, labels_HS, labels_SG, labels_avrg, imagename = validate_dataset[Image_index_to_demonstrate]
images = np.mean(images, axis=0)
i_height, i_width = images.shape
print(images.shape)
# print('The dimension of image, channel:' + str(np.shape(images)[0]) + ', height:' + str(np.shape(images)[1]) + ', width:' + str(np.shape(images)[2]))
# print('The dimension of label, channel:' + str(np.shape(labels_over)[0]) + ', height:' + str(np.shape(labels_over)[1]) + ', width:' + str(np.shape(labels_over)[2]))

# plot input image:
# the input image is original mnist images with gaussian noises
# plt.imshow(np.mean(images, axis=0), cmap='gray')
# plt.title('Input image')
# plt.show()

# plot the labels:
fig = plt.figure(figsize=(9, 13))
#columns = 4 # images, labels_AR, labels_HS, labels_SG
columns = 5
rows = 1
ax = []
labels = []
labels_names = []
labels.append(images)
labels.append(labels_AR)
labels.append(labels_HS)
labels.append(labels_SG)
labels.append(labels_avrg)
labels_names.append('Input')
labels_names.append('AR label')
labels_names.append('HS label')
labels_names.append('SG label')
labels_names.append('Average label')

for i in range(columns*rows):
    if i != 0:
        label_ = labels[i][0, :, :]
    else:
        label_ = labels[i]
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title(labels_names[i]) 
    plt.imshow(label_, cmap='gray')
plt.show()

# +
# ===== #
# Model
# ===== #

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# call model:
model = UNet_GlobalCMs(in_ch = input_dim, width = width, depth = depth, class_no = class_no, 
                       input_height = i_height, input_width = i_width, annotators = 4).to(device)

# model name for saving:
model_name = 'UNet_Confusion_Matrices_' + '_width' + str(width) + \
           '_depth' + str(depth) + '_train_batch_' + str(train_batchsize) + \
           '_alpha_' + str(alpha) + '_e' + str(num_epochs) + \
           '_lr' + str(learning_rate) 

# setting up device:
model.to(device)

# +
# =================================================== #
# Prepare folders to save trained models and results 
# =================================================== #

# save location:
saved_information_path = './Results_Oocytes_Gent'
try:
    os.mkdir(saved_information_path)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

saved_information_path = saved_information_path + '/' + model_name
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

save_path_visual_result = saved_information_path + '/visual_results'
try:
    os.mkdir(save_path_visual_result)
except OSError as exc:
    if exc.errno != errno.EEXIST:
        raise
    pass

# tensorboardX file saved location:
writer = SummaryWriter('./Results_Oocytes_Gent/Log_' + model_name)

# +
# =================================================== #
# Training
# =================================================== #

# We use adamW optimiser for more accurate L2 regularisation
optimizer = AdamW(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-8, weight_decay=1e-5)

for epoch in range(num_epochs):
    model.train()
    
    running_loss = 0
    running_loss_ce = 0
    running_loss_trace = 0
    running_iou = 0
        
    print(device)
    
    for j, (images, labels_AR, labels_HS, labels_SG, labels_avrg, imagename) in enumerate(trainloader):
        
        b, c, h, w = images.size()

        # zero gradients before each iteration
        optimizer.zero_grad()
        
        # cast numpy data into tensor float
        images = images.to(device=device, dtype=torch.float32)
        labels_AR = labels_AR.to(device=device, dtype=torch.float32)
        labels_HS = labels_HS.to(device=device, dtype=torch.float32)
        labels_SG = labels_SG.to(device=device, dtype=torch.float32)
        labels_avrg = labels_avrg.to(device=device, dtype=torch.float32)
        
        labels_all = []
        labels_all.append(labels_AR)
        labels_all.append(labels_HS)
        labels_all.append(labels_SG)
        labels_all.append(labels_avrg)
        
        # model has two outputs: 
        # first one is the probability map for true ground truth 
        # second one is a list collection of probability maps for different noisy ground truths
        
        outputs_logits, outputs_logits_noisy = model(images)
        
        # calculate loss:
        # loss: total loss
        # loss_ce: main cross entropy loss
        # loss_trace: regularisation loss
        loss, loss_ce, loss_trace = noisy_label_loss(outputs_logits, outputs_logits_noisy, labels_all, alpha)

        # calculate the gradients:
        loss.backward()
        # update weights in model:
        optimizer.step()
        
        _, train_output = torch.max(outputs_logits, dim=1)
        train_iou = segmentation_scores(labels_avrg.cpu().detach().numpy(), train_output.cpu().detach().numpy(), class_no)
        running_loss += loss
        running_loss_ce += loss_ce
        running_loss_trace += loss_trace
        running_iou += train_iou

        if (j + 1) == 1:
            # check the validation accuray at the begning of each epoch:
            v_dice, v_ged = evaluate_noisy_label_4(data=validateloader,
                                                   model1=model,
                                                   class_no=class_no)
            
            print(
                'Step [{}/{}], '
                'Val dice: {:.4f},'
                'Val GED: {:.4f},'
                'loss main: {:.4f},'
                'loss regualrisation: {:.4f},'.format(epoch + 1, num_epochs,
                                                            v_dice,
                                                            v_ged,
                                                            running_loss_ce / (j + 1),
                                                            running_loss_trace / (j + 1)))
        
            writer.add_scalars('scalars', {'loss': running_loss / (j + 1),
                                           'train iou': running_iou / (j + 1),
                                           'val iou': v_dice,
                                           'train main loss': running_loss_ce / (j + 1),
                                           'train regularisation loss': running_loss_trace / (j + 1)}, epoch + 1)

# save model:
save_model_name_full = saved_model_path + '/' + model_name + '_Final.pt'
torch.save(model, save_model_name_full)
print('\n')
print('Training ended')
# -

# =================================================== #
# Testing
# =================================================== #
model.eval()
for i, (v_images, labels_AR, labels_HS, labels_SG, labels_avrg, imagename) in enumerate(testloader):
        v_images = v_images.to(device=device, dtype=torch.float32)
        v_outputs_logits_original, v_outputs_logits_noisy = model(v_images)
        b, c, h, w = v_outputs_logits_original.size()
        # plot the final segmentation map
        v_outputs_logits_original = nn.Softmax(dim=1)(v_outputs_logits_original)
        _, v_outputs_logits = torch.max(v_outputs_logits_original, dim=1)

        save_name = save_path_visual_result + '/test_' + str(i) + '_seg.png'
        save_name_label = save_path_visual_result + '/test_' + str(i) + '_label.png'
        save_name_slice = save_path_visual_result + '/test_' + str(i) + '_img.png'

        plt.imsave(save_name_slice, v_images[:, 1, :, :].reshape(h, w).cpu().detach().numpy(), cmap='gray')
        plt.imsave(save_name, v_outputs_logits.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        plt.imsave(save_name_label, labels_avrg.reshape(h, w).cpu().detach().numpy(), cmap='gray')
        
        # plot the noisy segmentation maps:
        v_outputs_logits_original = v_outputs_logits_original.reshape(b, c, h*w)
        v_outputs_logits_original = v_outputs_logits_original.permute(0, 2, 1).contiguous()
        v_outputs_logits_original = v_outputs_logits_original.view(b * h * w, c).view(b*h*w, c, 1)
        for j, cm in enumerate(v_outputs_logits_noisy):
            cm = cm.view(b, c**2, h*w).permute(0, 2, 1).contiguous().view(b*h*w, c*c).view(b*h*w, c, c)
            cm = cm / cm.sum(1, keepdim=True)
            v_noisy_output_original = torch.bmm(cm, v_outputs_logits_original).view(b*h*w, c)
            v_noisy_output_original = v_noisy_output_original.view(b, h*w, c).permute(0, 2, 1).contiguous().view(b, c, h, w)
            _, v_noisy_output = torch.max(v_noisy_output_original, dim=1)
            save_name = save_path_visual_result + '/test_' + str(i) + '_noisy_' + str(j) + '_seg.png'
            print(save_name)
            plt.imsave(save_name, v_noisy_output.reshape(h, w).cpu().detach().numpy(), cmap='gray')

# +
# =================================================== #
# Predictions Plot
# =================================================== #
test_data_index = 3

AR_seg = save_path_visual_result + '/test_' + str(test_data_index) + '_noisy_' + str(0) + '_seg.png'
HS_seg = save_path_visual_result + '/test_' + str(test_data_index) + '_noisy_' + str(1) + '_seg.png'
SG_seg = save_path_visual_result + '/test_' + str(test_data_index) + '_noisy_' + str(2) + '_seg.png'
avrg_seg = save_path_visual_result + '/test_' + str(test_data_index) + '_noisy_' + str(3) + '_seg.png'


seg = save_path_visual_result + '/test_' + str(test_data_index) + '_seg.png'
label = save_path_visual_result + '/test_' + str(test_data_index) + '_label.png'
img = save_path_visual_result + '/test_' + str(test_data_index) + '_img.png'

# plot image, ground truth and final segmentation
fig = plt.figure(figsize=(6.7, 13))
columns = 3
rows = 1

ax = []
imgs = []
imgs_names = []

imgs.append(img)
imgs.append(label)
imgs.append(seg)

imgs_names.append('Test img')
imgs_names.append('GroundTruth')
imgs_names.append('Pred of true seg')

for i in range(columns*rows):
    img_ = imgs[i]
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title(imgs_names[i]) 
    img_ = Image.open(img_)
    img_ = np.array(img_, dtype='uint8')
    plt.imshow(img_, cmap='gray')
plt.show()

# plot the segmentation for noisy labels:
fig = plt.figure(figsize=(9, 13))
columns = 4
rows = 1

ax = []
noisy_segs = []
noisy_segs_names = []

noisy_segs.append(AR_seg)
noisy_segs.append(HS_seg)
noisy_segs.append(SG_seg)
noisy_segs.append(avrg_seg)

noisy_segs_names.append('Pred of AR')
noisy_segs_names.append('Pred of HS')
noisy_segs_names.append('Pred of SG')
noisy_segs_names.append('Pred of avrg')

for i in range(columns*rows):
    noisy_seg_ = noisy_segs[i]
    ax.append(fig.add_subplot(rows, columns, i+1))
    ax[-1].set_title(noisy_segs_names[i]) 
    noisy_seg_ = Image.open(noisy_seg_)
    noisy_seg_ = np.array(noisy_seg_, dtype='uint8' )
    plt.imshow(noisy_seg_, cmap='gray')
plt.show()
# -


