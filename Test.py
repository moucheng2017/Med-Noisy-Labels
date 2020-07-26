import torch
# import sys
# sys.path.append("..")
from NNTrain_2 import BaselineMode
from NNTrain_6 import trainTwoModels
from NNTrain_5 import trainModels
from NNTrain_punet import train_punet
# torch.manual_seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
# ====================================

if __name__ == '__main__':
    #
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
    # regularisation_type: "1" is the default, which is the trace.mean()
    # alpha: weight of regularisation, it should be larger than 0.5, default value is 1
    # ==========================================
    # ==========================================
    # To use normal confusion matrices:
    # low_rank_mode = False,
    # rank = 0
    # To use low-rank approximation:
    # low_rank_mode = True
    # rank = r << class_no
    # =====================
    #
    # epoch_threshold: where we switch on the main loss, before threshold, main loss is zero
    # alpha_initial: initial weighting for trace regularisation
    # regularisation_type:
    #   (1): trace(cm)
    #   (2): trace(cm_transpose)
    #   (3): trace(cm) + trace(cm_transpose)
    #
    # trainTwoModels(input_dim=4,
    #                class_no=4,
    #                repeat=1,
    #                train_batchsize=2,
    #                validate_batchsize=1,
    #                num_epochs=30,
    #                learning_rate=1e-4,
    #                alpha=1.5,
    #                width=16,
    #                depth=4,
    #                data_path='/home/moucheng/Desktop/All_L0_H10',
    #                dataset_tag='brats',
    #                label_mode='multi',
    #                save_probability_map=True,
    #                low_rank_mode=False,
    #                rank=0,
    #                epoch_threshold=0,
    #                alpha_initial=-1.5,
    #                regularisation_type='2')
    #
    trainModels(input_dim=4,
                class_no=4,
                repeat=1,
                train_batchsize=2,
                validate_batchsize=1,
                num_epochs=20,
                learning_rate=1e-2,
                alpha=0.4,
                width=32,
                depth=4,
                data_path='/home/moucheng/Desktop/All_L0_H10',
                dataset_tag='brats',
                label_mode='multi',
                save_probability_map=True,
                low_rank_mode=True,
                rank=1)
    #
    # ============================================
    # ============================================
    # for baseline:
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
    # # #

    #

