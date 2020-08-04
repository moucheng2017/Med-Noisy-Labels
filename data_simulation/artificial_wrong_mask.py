from __future__ import print_function, division

import logging
import os
import signal
import time
import numpy as np
from nibabel import load as load_nii
import nibabel as nib



import SimpleITK as sitk
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


# def zscore_normalize(img, mask=None):
#     """
#     normalize a target image by subtracting the mean of the whole brain
#     and dividing by the standard deviation
#     Args:
#         img (nibabel.nifti1.Nifti1Image): target MR brain image
#         mask (nibabel.nifti1.Nifti1Image): brain mask for img
#     Returns:
#         normalized (nibabel.nifti1.Nifti1Image): img with WM mean at norm_value
#     """

#     img_data = img.get_data()
#     if mask is not None and not isinstance(mask, str):
#         mask_data = mask.get_data()
#     elif mask == 'nomask':
#         mask_data = img_data == img_data
#     else:
#         mask_data = img_data > img_data.mean()
#     logical_mask = mask_data == 1  # force the mask to be logical type
#     mean = img_data[logical_mask].mean()
#     std = img_data[logical_mask].std()
#     normalized = nib.Nifti1Image((img_data - mean) / std, img.affine, img.header)
#     return normalized

# def replaceRandom(arr, num):
#     temp = np.asarray(arr)   # Cast to numpy array
#     shape = temp.shape       # Store original shape
#     temp = temp.flatten()    # Flatten to 1D
#     inds = np.random.choice(temp.size, size=num)   # Get random indices
#     temp[inds] = np.random.normal(size=num)        # Fill with something
#     temp = temp.reshape(shape)                     # Restore original shape
#     return temp

for ii in range(1, 11):
    file_path = '/media/le/Local Disk/BDMS/ISBI2015/Data/Training_MS'
    file_folder = '%s/' % ii
    file_name = 'mask.nii.gz'
    images_path = os.path.join(file_path, file_folder, file_name)
    print(images_path)

    images = nib.load(images_path)
    # print('images', images)

    # print ('images_norm', images)
    ###########
    #
    # output_sequence = nib.Nifti1Image(images_norm, affine=images.affine)
    # output_sequence.to_filename('images_norm.nii.gz')

    # images_norm = zscore_normalize(images, mask=None)
    # >>> import nibabel as nib
    # >>> epi_img = nib.load('downloads/someones_epi.nii.gz')
    # >>> epi_img_data = epi_img.get_fdata()

    #print (images.shape)
    #

    this = images.get_data()
    img_data_arr = np.asarray(this)

    # print (img_data_arr)
    # print (np.nonzero(img_data_arr))
    # print (np.nonzero(img_data_arr)[0][:10])
    # print (len(np.nonzero(img_data_arr)[0]))
    # print (len(np.nonzero(img_data_arr)[1]))
    # print (img_data_arr.dtype)
    # img_data_arr[np.nonzero(img_data_arr)] =  np.random.randint(2, size=len(img_data_arr[np.nonzero(img_data_arr)]))
    arr = np.ones(len(img_data_arr[np.nonzero(img_data_arr)]))
    # print (arr[0:100])
    for i in range(0,len(img_data_arr[np.nonzero(img_data_arr)]), 100):
        # img_data_arr[np.nonzero(img_data_arr)][i:i+100]= np.random.randint(2, size=len(img_data_arr[np.nonzero(img_data_arr)][i:i+100]))
        # np.where(img_data_arr[np.nonzero(img_data_arr)][i:i+100] ,np.random.randint(2, size=len(img_data_arr[np.nonzero(img_data_arr)][i:i+100])))
        # arr = img_data_arr[np.nonzero(img_data_arr)][i:i+100]
        # rand = np.random.randint(2, size=1)
        # arr = img_data_arr[np.nonzero(img_data_arr)][i:i+100]
        # np.place(arr, arr==1, 0)
        # print (arr.dtype)
        # img_data_arr[np.nonzero(img_data_arr)][i:i+100] = arr
        arr[i*2:i*2 + 100] = 0
        # arr[i:i+50] = 0
        # arr[i:i+100] = 0

        # print (arr[i:i+100])
        # print (arr)

    ## out_arr = x[np.nonzero(x)]
    #shape = img_data_arr.shape       # Store original shape
    #temp = img_data_arr.flatten()    # Flatten to 1D
    #temp = np.random.shuffle(temp)   # Get random indices
    #print(temp)
    #       # Fill with something
    #img_data_this = np.random.shuffle(img_data_arr)
    #
    #
    #  print (arr[np.nonzero(arr)])
    img_data_arr[np.nonzero(img_data_arr)] = arr
    #  print (np.nonzero(img_data_arr))

    ni_img = nib.Nifti1Image(img_data_arr, images.affine, images.header)

    save_file_path = '/media/le/Local Disk/BDMS/ISBI2015/Data/Training_MS'
    save_file_folder = '%s/' % ii
    save_file_name = 'mask_wrong_2.nii.gz'
    save_images_path = os.path.join(save_file_path, save_file_folder, save_file_name)
    print(save_images_path)

    nib.save(ni_img, save_images_path)
    # ni_img.to_filename(os.path.join(save_file_path, save_file_folder, save_file_name))

    # os.chdir("/media/le/Local Disk/BDMS/MNIST/Nifity/")
    # os.system('fslmaths ./Wrong/${ii}.nii.gz -edge ./Wrong_Blur/${ii}.nii.gz')
    # os.system('fslmaths ./Wrong/${ii}.nii.gz -add ./Wrong_Blur/${ii}.nii.gz -bin ./Wrong_GN/${ii}')
    #
    # os.system('fslmaths ./Wrong_GN/${ii}.nii.gz -edge ./Wrong_Blur/${ii}.nii.gz')
    # os.system('fslmaths ./Wrong_GN/${ii}.nii.gz -add ./Wrong_Blur/${ii}.nii.gz -bin ./Wrong_GN/${ii}')

    ## print (img_data_arr)

    # images_norm.to_filename(os.path.join('/home/le/Desktop/working folder/25-002-bl_training/25-002-bl-t2_lesions_t13d_space.nii.gz'))





    #url = '/home/le/Desktop/working folder/25-003-bl_training/25-003-bl-flair_t13d_space_norm.nii.gz'
    #image = sitk.ReadImage(url)
    #result = sitk.GetArrayFromImage(image)
    #print(type(result))
    #print(result.shape)
    #plt.figure('historgram')
    #result = result.flatten()
    #n, bins, patches = plt.hist(result, bins=256, range= (1,result.max()),normed=0, facecolor='red', alpha=0.75,histtype = 'step')
    #plt.show()
