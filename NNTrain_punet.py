import torch
import os
import errno
import random
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
# from load_LIDC_data import LIDC_IDRI
from probabilistic_unet import ProbabilisticUnet

from NNUtils import CustomDataset_punet, test_punet, evaluate_punet
from NNMetrics import generalized_energy_distance, segmentation_scores

# ===================
# main computation:
# ===================


def train_punet(epochs,
                iteration,
                train_batch_size,
                lr,
                num_filters,
                input_channels,
                latent_dim,
                no_conv_fcomb,
                num_classes,
                beta,
                test_samples_no,
                dataset_path,
                dataset_tag):

    for itr in range(iteration):

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        train_path = dataset_path + '/train'
        validate_path = dataset_path + '/validate'
        test_path = dataset_path + '/test'

        dataset_train = CustomDataset_punet(dataset_location=train_path, dataset_tag=dataset_tag, noisylabel='p_unet', augmentation=True)
        dataset_val = CustomDataset_punet(dataset_location=validate_path, dataset_tag=dataset_tag, noisylabel='multi', augmentation=False)
        dataset_test = CustomDataset_punet(dataset_location=test_path, dataset_tag=dataset_tag, noisylabel='multi', augmentation=False)
        # dataset_size = len(dataset_train)

        # indices = list(range(dataset_size))
        # split = int(np.floor(0.1 * dataset_size))
        # np.random.shuffle(indices)
        # train_indices, test_indices = indices[split:], indices[:split]
        # train_sampler = SubsetRandomSampler(train_indices)
        # test_sampler = SubsetRandomSampler(test_indices)
        # print("Number of training/test patches:", (len(train_indices),len(test_indices)))

        train_loader = DataLoader(dataset_train, batch_size=train_batch_size, shuffle=True, num_workers=4, drop_last=True)

        val_loader = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        test_loader = DataLoader(dataset_test, batch_size=1, shuffle=False, num_workers=1, drop_last=False)

        # net = ProbabilisticUnet(input_channels=3, num_classes=1, num_filters=[8, 16, 32, 64], latent_dim=4, no_convs_fcomb=2, beta=10)
        net = ProbabilisticUnet(input_channels=input_channels, num_classes=num_classes, num_filters=num_filters, latent_dim=latent_dim, no_convs_fcomb=no_conv_fcomb, beta=beta)

        net.to(device)

        optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=1e-5)

        # epochs = 100

        training_iterations = len(dataset_train) // train_batch_size - 1

        for epoch in range(epochs):
            #
            net.train()
            #
            for step, (patch, mask, mask_name) in enumerate(train_loader):
                #
                # mask_list = [mask_over, mask_under, mask_wrong, mask_true]
                # mask = random.choice(mask_list)
                # print(np.unique(mask))
                #
                patch = patch.to(device)
                mask = mask.to(device)
                # mask = torch.unsqueeze(mask,1)
                net.forward(patch, mask, training=True)
                elbo, reconstruction, kl = net.elbo(mask)
                # reg_loss = l2_regularisation(net.posterior) + l2_regularisation(net.prior) + l2_regularisation(net.fcomb.layers)
                # loss = -elbo + 1e-5 * reg_loss
                loss = -elbo
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                #
                epoch_noisy_labels = []
                epoch_noisy_segs = []
                #
                if (step + 1) == training_iterations:
                    #
                    validate_iou = 0
                    generalized_energy_distance_epoch = 0
                    #
                    validate_iou, generalized_energy_distance_epoch = evaluate_punet(net=net, val_data=val_loader, class_no=num_classes, sampling_no=4)
                    print('epoch:' + str(epoch))
                    print('val dice: ' + str(validate_iou))
                    print('val generalized_energy: ' + str(generalized_energy_distance_epoch))
                    print('train loss: ' + str(loss.item()))
                    print('kl is: ' + str(kl.item()))
                    print('reconstruction loss is: ' + str(reconstruction.item()))
                    print('\n')
        #
        print('\n')
        #
        save_path = '../Exp_Results_PUnet'
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
        save_path = save_path + '/Exp_' + str(itr) + \
                    '_punet_' + \
                    '_train_batch_' + str(train_batch_size) + \
                    '_latent_dim_' + str(latent_dim) + \
                    '_lr_' + str(lr) + \
                    '_epochs_' + str(epochs) + \
                    '_beta_' + str(beta) + \
                    '_test_sample_no_' + str(test_samples_no)
        #
        test_punet(net=net, testdata=test_loader, save_path=save_path, sampling_times=test_samples_no)
        #
    print('Training is finished.')
#




