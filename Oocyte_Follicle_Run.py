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

    trainUnet(dataset_tag = 'oocytes_cph',
                dataset_name = 'oocytes_cph',
                data_directory = '/data/eurova/multi_annotators_project/LNLMI/oocytes_cph/',
                input_dim = 3,
                class_no = 2,
                repeat = 1,
                train_batchsize = 2,
                validate_batchsize = 1,
                num_epochs = 200,
                learning_rate = 1e-3,
                width = 32,
                depth = 5,
                augmentation='all_flip',
                loss_f='dice',
                path_name = '/data/eurova/oocyte_follicle_database/torch/Results/' + time.strftime("%Y%m%d-%H%M%S"),
                labels_mode = 'avrg')