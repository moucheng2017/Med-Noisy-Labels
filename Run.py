import torch
# import sys
# sys.path.append("..")
from Train_unet import trainUnet
from Train_ours import trainModels
from Train_GCM import trainGCMModels
from Train_unet import trainUnet
from Train_step_CMs import trainStepCM
# from Train_punet import train_punet
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
import time
# ====================================

if __name__ == '__main__':
    # ========================================================
    # comment out other train functions when u use one of them
    # for noisy labels with learnt confusion matrices:
    # input_dim: number of channels of input, e.g. 3 for RGB; 1 for CT
    # class_no: class number
    # repeat: how many times do you repeat the same experiment
    # depth: depth of network, of down-samping stages
    # width: number of channels in first encoder in network, the number of channels is doubled in each down-sampling stage
    # train_batchsize:
    # validate batchsize
    # num_epochs: number of epochs
    # learning rate
    # data_path: path to your training data
    # dataset_tag: "mnist" for MNIST digits; "brats" for BRATS; "lidc" for LIDC lung nodule datasets with 4 annotators
    # label_mode: "multi" for noisy_labels
    # alpha: weight of regularisation, it should be larger than 0.5, default value is 1
    # ==========================================
    # for training with our model
    # ==========================================
    # trainModels(input_dim=3,
    #             class_no=2,
    #             repeat=1,
    #             train_batchsize=8,
    #             validate_batchsize=1,
    #             num_epochs=20,
    #             learning_rate=1e-4,
    #             alpha=0.001,
    #             width=32,
    #             depth=5,
    #             data_path='/data/eurova/multi_annotators_project/LNLMI/oocytes_gent/',
    #             dataset_tag='oocytes_gent',
    #             label_mode='multi',
    #             save_probability_map=False,
    #             low_rank_mode=False,
    #             path_name = '/data/eurova/multi_annotators_project/LNLMI/Results/CMs_Results/' + time.strftime("%Y%m%d-%H%M%S"))
    
    # ============================================
    # for baseline with global confusion  matrices
    # ============================================
    # trainGCMModels(input_dim=3,
    #                class_no=2,
    #                repeat=1,
    #                train_batchsize=32,
    #                validate_batchsize=1,
    #                num_epochs=30,
    #                learning_rate=1e-4,
    #                input_height=192,
    #                input_width=256,
    #                alpha=0.01,
    #                width=32,
    #                depth=5,
    #                data_path='/data/eurova/multi_annotators_project/LNLMI/oocytes_gent/',
    #                dataset_tag='oocytes_gent',
    #                label_mode='multi',
    #                loss_f='noisy_label',
    #                save_probability_map=False,
    #                path_name = '/data/eurova/multi_annotators_project/LNLMI/Results/Global_CMs_Results/' + time.strftime("%Y%m%d-%H%M%S"))
    # ============================================
    # for baseline without label merging:
    # ============================================
    # BaselineMode(input_dim=1,
    #              class_no=4,
    #              repeat=2,
    #              train_batchsize=16,
    #              validate_batchsize=1,
    #              num_epochs=1,
    #              learning_rate=1e-4,
    #              width=16,
    #              depth=3,
    #              network='unet',
    #              dataset_location='/home/moucheng/Desktop/brats',
    #              dataset_tag='brats',
    #              loss_f='ce')
    # ============================================
    # for probabilistic u-net
    # ============================================
    # train_punet(epochs=80,
    #             iteration=3,
    #             train_batch_size=20,
    #             lr=1e-4,
    #             num_filters=[32, 64, 128, 256],
    #             input_channels=3,
    #             latent_dim=6,
    #             no_conv_fcomb=2,
    #             num_classes=2,
    #             beta=5,
    #             test_samples_no=10,
    #             dataset_path='Path',
    #             dataset_tag='mnist')
    #
    # ============================================
    # for simple u-net
    # ============================================
    # trainUnet(dataset_tag = 'oocytes_gent',
    #             dataset_name = 'oocytes_gent',
    #             data_directory = '/data/eurova/multi_annotators_project/LNLMI/oocytes_gent/',
    #             input_dim = 3,
    #             class_no = 2,
    #             repeat = 1,
    #             train_batchsize = 2,
    #             validate_batchsize = 1,
    #             num_epochs = 200,
    #             learning_rate = 1e-3,
    #             width = 32,
    #             depth = 5,
    #             augmentation='all_flip',
    #             loss_f='dice',
    #             path_name = '/data/eurova/multi_annotators_project/LNLMI/Results/Maj_Results/' + time.strftime("%Y%m%d-%H%M%S"),
    #             labels_mode = 'avrg')
    #
    # ============================================
    # for simple u-net for skin training
    # ============================================
    trainUnet(dataset_tag = 'skin',
                dataset_name = 'skin',
                data_directory = '/data/eurova/multi_annotators_project/LNLMI/skin_np_192_256_3/',
                input_dim = 3,
                class_no = 2,
                repeat = 1,
                train_batchsize = 16,
                validate_batchsize = 1,
                num_epochs = 150,
                learning_rate = 1e-3,
                width = 32,
                depth = 5,
                augmentation='all_flip',
                loss_f='dice',
                path_name = '/data/eurova/multi_annotators_project/LNLMI/Results/skin/' + time.strftime("%Y%m%d-%H%M%S"),
                labels_mode = 'skin')
    # # #
    #
    # ============================================
    # Stepwise method: Step CMs
    # ============================================
    # trainStepCM(input_dim = 3,
    #             class_no = 2,
    #             repeat = 1,
    #             train_batchsize = 2,
    #             validate_batchsize = 1,
    #             num_epochs = 20,
    #             learning_rate = 1e-2,
    #             input_height = 192,
    #             input_width = 256,
    #             alpha = 10.0,
    #             width = 32,
    #             depth = 3,
    #             data_path = '/data/eurova/multi_annotators_project/LNLMI/oocytes_gent/',
    #             dataset_tag = 'oocytes_gent',
    #             label_mode = 'multi',
    #             loss_f = 'noisy_label',
    #             save_probability_map = True,
    #             path_name = '/data/eurova/multi_annotators_project/LNLMI/Results/Stepwise_Results/' + time.strftime("%Y%m%d-%H%M%S"))
    # # #
    #
