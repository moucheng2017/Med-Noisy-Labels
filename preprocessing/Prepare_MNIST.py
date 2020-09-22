import os
import gzip
import errno
import shutil
import random
# import pydicom
import numpy as np
from PIL import Image
import nibabel as nib
import matplotlib.pyplot as plt

from scipy.ndimage.morphology import binary_fill_holes
from skimage.transform import resize
# from nipype.interfaces.ants import N4BiasFieldCorrection
from tifffile import imsave


def chunks(l, n):
    # l: the whole list to be divided
    # n: amount of elements for each subgroup
    # Yield successive n-sized chunks from l
    for i in range(0, len(l), n):
        yield l[i:i + n]


def divide_data(data_folder):
    #
    all_cases = os.listdir(data_folder)
    all_cases = [os.path.join(data_folder, x) for index, x in enumerate(all_cases)]
    random.shuffle(all_cases)
    #
    all_cases = list(chunks(all_cases, len(all_cases) // 10))
    #
    all_test = all_cases[0] + all_cases[1]
    all_validate = all_cases[2]
    all_train = all_cases[3] + all_cases[4] + all_cases[5] + \
                all_cases[6] + all_cases[7] + all_cases[8] + \
                all_cases[9]

    return all_train, all_validate, all_test


def generate_patches(all_cases, save_path, tag):
    # - train
    #   - mean
    #   - gaussian
    #   - over
    #   - under
    # - validate
    #   - mean
    #   - gaussian
    #   - over
    #   - under
    # save mother folders:
    save_path = save_path + '/' + tag
    # selected modalities:
    try:
        os.makedirs(save_path)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass
    #
    for case in all_cases:
        #
        fullfilename, extenstion = os.path.splitext(case)
        dirpath_parts = fullfilename.split('/')
        case_index = dirpath_parts[-1]
        dirpath_parts = case_index.split('.')
        case_index = dirpath_parts[0]
        #
        print(case_index)
        #
        all_data = os.listdir(case)
        all_data = [os.path.join(case, x) for index, x in enumerate(all_data) if 'Mean.nii.gz' in x
                    or 'Over.nii.gz' in x
                    or 'Under.nii.gz' in x
                    or 'Wrong.nii.gz' in x
                    or 'GT.nii.gz' in x
                    or 'Gaussian.nii.gz' in x]
        #
        for data_path in all_data:
            #
            f_data = nib.load(data_path)
            #
            f_data = f_data.get_fdata()
            #
            f_data = np.asarray(f_data, dtype=np.float32)
            #
            if 'Gaussian.nii.gz' in data_path:
                f_data = f_data
            else:
                f_data = f_data[:, :, 1]
                unique, counts = np.unique(f_data, return_counts=True)
                if 'Mean' not in data_path:
                    if len(unique) != 2:
                        f_data = np.where(f_data > 40.0, 1.0, 0.0)
                        unique, counts = np.unique(f_data, return_counts=True)
                # print(len(unique))

            #
            data_dirpath_parts = data_path.split('/')
            modality = data_dirpath_parts[-1]
            modality_parts = modality.split('.')
            modality = modality_parts[0]
            save_path_specific = save_path + '/' + modality
            print(modality)
            print(f_data.shape)
            # print(np.unique(f_data))
            #
            try:
                #
                os.makedirs(save_path_specific)
                #
            except OSError as exc:
                #
                if exc.errno != errno.EEXIST:
                    #
                    raise
            pass
            #
            # plt.show()
            # plt.imshow(f_data)
            # print(np.unique(f_data))
            save_name = save_path_specific + '/' + case_index + '_' + modality + '.tif'
            imsave(save_name, f_data)
            print(case_index + '_' + modality + '.tif' + ' is saved')
            print('\n')
            #


def main_loop(data_folder, store_folder, tag):

    try:
        os.makedirs(store_folder)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    # train_cases, validate_cases, test_cases = divide_data(data_folder=data_folder)

    cases = [os.path.join(data_folder, x) for x in os.listdir(data_folder)]

    generate_patches(all_cases=cases, save_path=store_folder, tag=tag)

    # generate_patches(all_cases=validate_cases, save_path=store_folder, tag='validate')

    # generate_patches(all_cases=test_cases, save_path=store_folder, tag='test')


if __name__ == '__main__':
    #
    # data_folder = '/home/moucheng/projects_data/Testing_Le'
    # data_folder = '/home/moucheng/projects_data/Training_MNIST'
    # save_folder = '/home/moucheng/projects_data/MNIST_train'

    data_folder = '../data_examples/MNIST_training'
    save_folder = '../MNIST_samples/training'

    main_loop(data_folder, save_folder, tag='train')
    #
    data_folder = '/home/moucheng/projects_data/Training_MNIST'
    #
    main_loop(data_folder, save_folder, tag='test')
    #
    main_loop(data_folder, save_folder, tag='validate')
    #
print('End')