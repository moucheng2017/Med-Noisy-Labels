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


# ==============================================
# This is for pre-processing the BRATS 2018 data
# ==============================================


def chunks(l, n):
    # l: the whole list to be divided
    # n: amount of elements for each subgroup
    # Yield successive n-sized chunks from l
    for i in range(0, len(l), n):
        yield l[i:i + n]


# def check_single_dicom_slice(slice_path, mask_path):
    # slice_path = '/home/moucheng/projects data/Pulmonary data/ILD dataset/ild_data/ILD_DB_volumeROIs/53/CT-INSPIRIUM-7605/CT-7605-0004.dcm'
    # mask_path = '/home/moucheng/projects data/Pulmonary data/ILD dataset/ild_data/ILD_DB_volumeROIs/53/CT-INSPIRIUM-7605/roi_mask/roi_mask_7605_4.dcm'
  #  lung_slice = pydicom.dcmread(slice_path).pixel_array
  #  mask_slice = pydicom.dcmread(mask_path).pixel_array
  #  maximum_value = lung_slice.max()
  #  label_value = mask_slice.max()
  #  print("CT slice data type: " + str(lung_slice.dtype))
  #  print("Mask data type: " + str(mask_slice.dtype))
  #  print("Label: " + str(label_value))
  #  plt.imshow(lung_slice, cmap=plt.cm.gray)
  #  plt.show()
  #  plt.imshow(mask_slice, cmap=plt.cm.gray)
  #  plt.show()
    # tune the parameter here for better visulisation of mask overlayed on the slice
  #  overlay = lung_slice + 0.25 * maximum_value * mask_slice
  #  plt.imshow(overlay, cmap=plt.cm.gray)


def unzip_all(dirName):
    # unzip all files with extension as '.gz'
    listOfFile = os.listdir(dirName)
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        AllZips = os.listdir(fullPath)
        for Zip in AllZips:
            fullZipPath = os.path.join(fullPath, Zip)
            savePath = fullZipPath.replace('.gz', '')
            with gzip.open(fullZipPath, 'rb') as f_in:
                with open(savePath, 'wb') as f_out:
                    shutil.copyfileobj(f_in, f_out)
            print(savePath + ' is unzipped and saved')
    print('\n')
    print('\n')
    print('All done')


def delete_all(dirName):
    # unzip all files with extension as '.gz'
    listOfFile = os.listdir(dirName)
    # Iterate over all the entries
    for entry in listOfFile:
        fullPath = os.path.join(dirName, entry)
        AllZips = os.listdir(fullPath)
        for Zip in AllZips:
            fullZipPath = os.path.join(fullPath, Zip)
            if '.gz' in fullZipPath:
                os.remove(fullZipPath)
            print(fullZipPath + ' is deleted')
    print('\n')
    print('\n')
    print('All done')


def generate_patches(data_folder, save_folder_mother, new_size, save_everything, tag_class, tag_category):
    # data_folder: prepared source data folder
    # save_folder_mother: the save folder
    # group_number: the fold index for cross-validation
    # new_size: target size of images to be stored
    # save_everything; a flag to control whether we should save the data or not
    # case_folder:
    #    - t1, t2, t1ce, flair and seg
    for item in data_folder:
        print(item)
        print('\n')
        print('\n')
        all_sub_folders = os.listdir(item)
        all_modalities = [os.path.join(item, x) for index, x in enumerate(
            all_sub_folders) if 'seg' not in x]
        # force all in the same order: flair, t1, t1ce, t2
        all_modalities.sort()
        gt_path = [os.path.join(item, x) for index, x in enumerate(
            all_sub_folders) if 'seg' in x]
        #         for p in all_modalities_lgg: print (p)
        #         print('\n')
        # read all modalities for case for validation
        t1 = nib.load(all_modalities[0])
        t1 = t1.get_fdata()
        t2 = nib.load(all_modalities[1])
        t2 = t2.get_fdata()
        t1ce = nib.load(all_modalities[2])
        t1ce = t1ce.get_fdata()
        flair = nib.load(all_modalities[3])
        flair = flair.get_fdata()
        # normalise based on all non-zero elements:
        t1_non_zero = t1[np.nonzero(t1)]
        t2_non_zero = t2[np.nonzero(t2)]
        t1ce_non_zero = t1ce[np.nonzero(t1ce)]
        flair_non_zero = flair[np.nonzero(flair)]
        #
        t1 = (t1 - t1_non_zero.mean()) / t1_non_zero.std()
        t2 = (t2 - t2_non_zero.mean()) / t2_non_zero.std()
        t1ce = (t1ce - t1ce_non_zero.mean()) / t1ce_non_zero.std()
        flair = (flair - flair_non_zero.mean()) / flair_non_zero.std()
        # ground truth of lgg case for validation:
        gt_path = gt_path[0]
        gt = nib.load(gt_path)
        gt = gt.get_fdata()
        # merge labels:
        # necrotic core + non-enhancing tumour core (1)
        # edema (2)
        # enhancing tumour (4)
        #
        # unique, counts = np.unique(gt, return_counts=True)
        # print(np.asarray((unique, counts)).T)
        #
        if tag_class == 'WT':
            gt[gt == 0] = 0
            gt[gt == 1] = 1
            gt[gt == 2] = 1
            gt[gt == 4] = 1
        elif tag_class == 'ET':
            gt[gt == 0] = 0
            gt[gt == 1] = 0
            gt[gt == 2] = 0
            gt[gt == 4] = 1
        elif tag_class == 'TC':
            gt[gt == 0] = 0
            gt[gt == 1] = 1
            gt[gt == 2] = 0
            gt[gt == 4] = 1
        elif tag_class == 'All':
            gt[gt == 0] = 0
            gt[gt == 1] = 1
            gt[gt == 2] = 2
            gt[gt == 4] = 3
        #
        # unique, counts = np.unique(gt, return_counts=True)
        # print(np.asarray((unique, counts)).T)
        #
        height, width, slice_no = gt.shape
        # extract case number and name:
        fullfilename, extenstion = os.path.splitext(gt_path)
        dirpath_parts = fullfilename.split('/')
        case_index = dirpath_parts[-1]
        for no in range(slice_no):
            # create store names:
            gt_slice_store_name = case_index + '_gt_' + str(no) + '.tif'
            img_slice_store_name = case_index + '_slice_' + str(no) + '.tif'
            # gt_slice_store_name = case_index + '_gt_' + str(no)
            # img_slice_store_name = case_index + '_slice_' + str(no)
            # switch store path:
            label_store_path = save_folder_mother + '/' + tag_category + '/labels'
            patch_store_path = save_folder_mother + '/' + tag_category + '/patches'
            # if train is False:
            #     label_store_path = save_folder_mother + '/' + \
            #         str(group_number + 1) + '/validate/labels'
            #     patch_store_path = save_folder_mother + '/' + \
            #         str(group_number + 1) + '/validate/patches'
            # else:
            #     label_store_path = save_folder_mother + '/' + \
            #         str(group_number + 1) + '/train/labels'
            #     patch_store_path = save_folder_mother + '/' + \
            #         str(group_number + 1) + '/train/patches'
            #
            label_store_path_full = os.path.join(
                label_store_path, gt_slice_store_name)
            patch_store_path_full = os.path.join(
                patch_store_path, img_slice_store_name)
            # store ground truth patches:
            gt_slice = gt[:, :, no]
            # gt_slice = binary_fill_holes(gt_slice).astype(int)
            #
            h, w = np.shape(gt_slice)
            gt_slice = np.asarray(gt_slice, dtype=np.float32)
            left = int(np.ceil((w - new_size) / 2))
            right = w - int(np.floor((w - new_size) / 2))
            top = int(np.ceil((h - new_size) / 2))
            bottom = h - int(np.floor((h - new_size) / 2))
            gt_slice = gt_slice[top:bottom, left:right]
            #
            # unique, counts = np.unique(gt, return_counts=True)
            # print(np.asarray((unique, counts)).T)
            #
            # if save_everything is True:
            #     # im.save(label_store_path_full, "TIFF")
            #     imsave(label_store_path_full, gt_slice)
            #     print(gt_slice_store_name + ' ground truth is stored')
            # concatenate slices for image data:
            t1_slice = t1[:, :, no]
            t1_slice = np.asarray(t1_slice, dtype=np.float32)
            t1_slice = t1_slice[top:bottom, left:right]
            t1_slice = np.reshape(t1_slice, (1, new_size, new_size))
            #
            t2_slice = t2[:, :, no]
            t2_slice = np.asarray(t2_slice, dtype=np.float32)
            t2_slice = t2_slice[top:bottom, left:right]
            t2_slice = np.reshape(t2_slice, (1, new_size, new_size))
            #
            t1ce_slice = t1ce[:, :, no]
            t1ce_slice = np.asarray(t1ce_slice, dtype=np.float32)
            t1ce_slice = t1ce_slice[top:bottom, left:right]
            t1ce_slice = np.reshape(t1ce_slice, (1, new_size, new_size))
            #
            flair_slice = flair[:, :, no]
            flair_slice = np.asarray(flair_slice, dtype=np.float32)
            flair_slice = flair_slice[top:bottom, left:right]
            flair_slice = np.reshape(flair_slice, (1, new_size, new_size))
            #
            multi_modal_slice = t1_slice
            multi_modal_slice = np.concatenate(
                (multi_modal_slice, t2_slice), axis=0)
            multi_modal_slice = np.concatenate(
                (multi_modal_slice, t1ce_slice), axis=0)
            multi_modal_slice = np.concatenate(
                (multi_modal_slice, flair_slice), axis=0)
            #
            if save_everything is True:
                #
                if t1_slice.max() > 0 and t2_slice.max() > 0 and t1ce_slice.max() > 0 and flair_slice.max() > 0:
                    #
                    non_zero_gt = np.count_nonzero(gt_slice)
                    #
                    non_zero_slice = np.count_nonzero(t2_slice)
                    #
                    if tag_class == 'ET':
                        #
                        if non_zero_gt > 1:
                            #
                            imsave(patch_store_path_full, multi_modal_slice)
                            # np.save(patch_store_path_full, multi_modal_slice)
                            # print(img_slice_store_name + ' of ' + tag_category + ' of ' + tag_class + 'image slice is stored')
                            imsave(label_store_path_full, gt_slice)
                            # np.save(label_store_path_full, gt_slice)
                            # print(gt_slice_store_name + ' of ' + tag_category + ' of ' + tag_class + ' ground truth is stored')
                            #
                        elif non_zero_slice > 1000:
                            #
                            augmentation = random.random()
                            #
                            if augmentation > 0.5:
                                #
                                # this condition is because otherwise it will be way too many useless training samples (e.g. containing zero information) saved
                                imsave(patch_store_path_full, multi_modal_slice)
                                # np.save(patch_store_path_full, multi_modal_slice)
                                # print(img_slice_store_name + ' of ' + tag_category + ' of ' + tag_class + 'image slice is stored')
                                imsave(label_store_path_full, gt_slice)
                                # np.save(label_store_path_full, gt_slice)
                                # print(gt_slice_store_name + ' of ' + tag_category + ' of ' + tag_class + ' ground truth is stored')
                    else:
                        #
                        if non_zero_gt > 1:
                            #
                            imsave(patch_store_path_full, multi_modal_slice)
                            # np.save(patch_store_path_full, multi_modal_slice)
                            # print(img_slice_store_name + ' of ' + tag_category + ' of ' + tag_class + 'image slice is stored')
                            imsave(label_store_path_full, gt_slice)
                            # np.save(label_store_path_full, gt_slice)
                            #
                            unique, counts = np.unique(gt_slice, return_counts=True)
                            print(np.asarray((unique, counts)).T)
                            # print(gt_slice_store_name + ' of ' + tag_category + ' of ' + tag_class + ' ground truth is stored')

            # print('\n')


def prepare_data(data_folder, LGG_cases, HGG_cases, tag_class):

    save_folder_mother = data_folder + '/' + tag_class + '_L' + \
        str(LGG_cases) + '_H' + str(HGG_cases)

    try:
        os.makedirs(save_folder_mother)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    image_train = save_folder_mother + '/train/patches'
    label_train = save_folder_mother + '/train/labels'
    image_validate = save_folder_mother + '/validate/patches'
    label_validate = save_folder_mother + '/validate/labels'
    image_test = save_folder_mother + '/test/patches'
    label_test = save_folder_mother + '/test/labels'

    try:
        os.makedirs(image_train)
        os.makedirs(label_train)
        os.makedirs(image_validate)
        os.makedirs(label_validate)
        os.makedirs(image_test)
        os.makedirs(label_test)
    except OSError as exc:
        if exc.errno != errno.EEXIST:
            raise
    pass

    lowgrade_folder = data_folder + '/LGG'
    highgrade_folder = data_folder + '/HGG'
    all_lowgrade_cases = os.listdir(lowgrade_folder)
    all_highgrade_cases = os.listdir(highgrade_folder)
    # randonly pick up cases from low grade and high grade separately:
    random_lowgrade_cases = np.random.choice(
        all_lowgrade_cases, int(LGG_cases))
    random_lowgrade_cases = [os.path.join(lowgrade_folder, x) for index, x in enumerate(
        random_lowgrade_cases) if os.path.isdir(os.path.join(lowgrade_folder, x))]
    random_highgrade_cases = np.random.choice(
        all_highgrade_cases, int(HGG_cases))
    random_highgrade_cases = [os.path.join(highgrade_folder, x) for index, x in enumerate(
        random_highgrade_cases) if os.path.isdir(os.path.join(highgrade_folder, x))]

    fold_amount = 10

    if LGG_cases > 0 and HGG_cases == 0:
        subgroups_lowgrade = list(
            chunks(random_lowgrade_cases, len(random_lowgrade_cases) // fold_amount))
        return random_lowgrade_cases, subgroups_lowgrade, save_folder_mother
    elif HGG_cases > 0 and LGG_cases == 0:
        subgroups_highgrade = list(
            chunks(random_highgrade_cases, len(random_highgrade_cases) // fold_amount))
        return random_highgrade_cases, subgroups_highgrade, save_folder_mother
    elif HGG_cases > 0 and LGG_cases > 0:
        subgroups_lowgrade = list(
            chunks(random_lowgrade_cases, len(random_lowgrade_cases) // fold_amount))
        subgroups_highgrade = list(
            chunks(random_highgrade_cases, len(random_highgrade_cases) // fold_amount))
        return random_highgrade_cases, random_lowgrade_cases, subgroups_highgrade, subgroups_lowgrade, save_folder_mother


def single_loop(total_cases, validate_cases_groups, save_folder_mother, new_size, tag_class):
    # for group_number, sub_group in enumerate(validate_cases_groups):

    validation = validate_cases_groups[0]
    test = validate_cases_groups[1] + validate_cases_groups[2]
    training = list(set(total_cases) - set(validation) - set(test))

    # print(training)

    generate_patches(validation, save_folder_mother, new_size, save_everything=True, tag_class=tag_class, tag_category='validate')

    generate_patches(training, save_folder_mother, new_size, save_everything=True, tag_class=tag_class, tag_category='train')

    generate_patches(test, save_folder_mother, new_size, save_everything=True, tag_class=tag_class, tag_category='test')


def main_loop(data_folder, lgg_amount, hgg_amount, new_size, tag_class):

    if lgg_amount > 0 and hgg_amount == 0:
        total_lowgrade_cases, subgroups_lowgrade, save_folder_mother = prepare_data(
            data_folder, LGG_cases=lgg_amount, HGG_cases=hgg_amount, tag_class=tag_class)
        single_loop(total_lowgrade_cases, subgroups_lowgrade,
                    save_folder_mother, new_size, tag_class=tag_class)
    elif hgg_amount > 0 and lgg_amount == 0:
        total_highgrade_cases, subgroups_highgrade, save_folder_mother = prepare_data(
            data_folder, LGG_cases=lgg_amount, HGG_cases=hgg_amount, tag_class=tag_class)
        single_loop(total_highgrade_cases, subgroups_highgrade,
                    save_folder_mother, new_size, tag_class=tag_class)
    elif hgg_amount > 0 and lgg_amount > 0:
        total_highgrade_cases, total_lowgrade_cases, subgroups_highgrade, subgroups_lowgrade, save_folder_mother = prepare_data(
            data_folder, LGG_cases=lgg_amount, HGG_cases=hgg_amount, tag_class=tag_class)
        single_loop(total_lowgrade_cases, subgroups_lowgrade,
                    save_folder_mother, new_size, tag_class=tag_class)
        single_loop(total_highgrade_cases, subgroups_highgrade,
                    save_folder_mother, new_size, tag_class=tag_class)


if __name__ == '__main__':
    # data_folder = '/cluster/project0/BRATS_2018_AuxiliaryNetwork/BRATS/MICCAI_BraTS_2018_Data_Training'
    # data_folder = '/home/moucheng/projects_data/Brain_data/BRATS2018/MICCAI_BraTS_2018_Data_Training'
    data_folder = '/home/moucheng/projects_data/Brain_data/BRATS2018/MICCAI_BraTS_2018_Data_Training'
    # hgg cases: 210 in total
    # lgg cases: 76 in total
    # original resolution: 240 x 240
    # tag:
    # ET: enhancing tumour (label 4)
    # WT: whole tumour (label 1 + label 2 + label 4)
    # TC: Tumour core (label 1 + label 4)
    #
    # For ET, only store images with brain araes larger than 1000 pixels at 50% change
    main_loop(data_folder, lgg_amount=10, hgg_amount=10, new_size=192, tag_class='All')
    #
print('End')
