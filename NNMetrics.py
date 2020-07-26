import torch
import torch.nn as nn
import numpy as np
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import _ni_support

from sklearn.metrics import confusion_matrix
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from scipy import ndimage
# good references:
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/utilities/util_common.py
# https://cmiclab.cs.ucl.ac.uk/CMIC/NiftyNet/blob/dev/niftynet/evaluation/pairwise_measures.py
# https://github.com/wkentaro/pytorch-fcn/blob/master/torchfcn/utils.py
# https://github.com/ozan-oktay/Attention-Gated-Networks/blob/master/utils/metrics.py
# https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/metrics.py
# -------------------------------------------------------------------------------


def calculate_cm(pred, true):
    #
    pred = pred.view(-1)
    true = true.view(-1)
    #
    pred = pred.cpu().detach().numpy()
    true = true.cpu().detach().numpy()
    #
    confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all')
    #
    # if tag == 'brats':
    #     confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all', labels=[0, 1, 2, 3])
    # else:
    #     confusion_matrices = confusion_matrix(y_true=true, y_pred=pred, normalize='all', labels=[0, 1])
    #
    #
    return confusion_matrices


def segmentation_scores(label_trues, label_preds, n_class):

    assert len(label_trues) == len(label_preds)

    if n_class == 2:
        #
        output_zeros = np.zeros_like(label_preds)
        output_ones = np.ones_like(label_preds)
        label_preds = np.where((label_preds > 0.5), output_ones, output_zeros)

    label_trues += 1
    label_preds += 1

    label_preds = np.asarray(label_preds, dtype='int8').copy()
    label_trues = np.asarray(label_trues, dtype='int8').copy()
    label_preds = label_preds * (label_trues > 0)

    intersection = label_preds * (label_preds == label_trues)
    (area_intersection, _) = np.histogram(intersection, bins=n_class, range=(1, n_class))
    (area_pred, _) = np.histogram(label_preds, bins=n_class, range=(1, n_class))
    (area_lab, _) = np.histogram(label_trues, bins=n_class, range=(1, n_class))
    # area_union = area_pred + area_lab - area_intersection
    area_union = area_pred + area_lab
    #
    return ((2 * area_intersection + 1e-6) / (area_union + 1e-6)).mean()


def generalized_energy_distance(all_gts, all_segs, class_no):
    gt_gt_dist = [segmentation_scores(gt_1, gt_2, class_no) for i1, gt_1 in enumerate(all_gts) for i2, gt_2 in enumerate(all_gts) if i1 != i2]
    seg_seg_dist = [segmentation_scores(seg_1, seg_2, class_no) for i1, seg_1 in enumerate(all_segs) for i2, seg_2 in enumerate(all_segs) if i1 != i2]
    seg_gt_list = [segmentation_scores(seg_, gt_, class_no) for i, seg_ in enumerate(all_segs) for j, gt_ in enumerate(all_gts)]
    ged_metric = sum(gt_gt_dist) / len(gt_gt_dist) + sum(seg_seg_dist) / len(seg_seg_dist) + 2 * sum(seg_gt_list) / len(seg_gt_list)
    return ged_metric

# # =======================================================================
# # reference :http://loli.github.io/medpy/_modules/medpy/metric/binary.html


def preprocessing_accuracy(label_true, label_pred, n_class):
    #
    if n_class == 2:
        output_zeros = np.zeros_like(label_pred)
        output_ones = np.ones_like(label_pred)
        label_pred = np.where((label_pred > 0.5), output_ones, output_zeros)
    #
    label_pred = np.asarray(label_pred, dtype='int8')
    label_true = np.asarray(label_true, dtype='int8')

    mask = (label_true >= 0) & (label_true < n_class) & (label_true != 8)

    label_true = label_true[mask].astype(int)
    label_pred = label_pred[mask].astype(int)

    return label_true, label_pred
#
#
def __surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result, reference = preprocessing_accuracy(reference, result)
    # reference = reference.cpu().detach().numpy()
    # result = result.cpu().detach().numpy()
    #
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]

    return sds


def hd95(result, reference, voxelspacing=None, connectivity=1):
    """
    95th percentile of the Hausdorff Distance.

    Computes the 95th percentile of the (symmetric) Hausdorff Distance (HD) between the binary objects in two
    images. Compared to the Hausdorff Distance, this metric is slightly more stable to small outliers and is
    commonly used in Biomedical Segmentation challenges.

    Parameters
    ----------
    result : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    reference : array_like
        Input data containing objects. Can be any type but will be converted
        into binary: background where 0, object everywhere else.
    voxelspacing : float or sequence of floats, optional
        The voxelspacing in a distance unit i.e. spacing of elements
        along each dimension. If a sequence, must be of length equal to
        the input rank; if a single number, this is used for all axes. If
        not specified, a grid spacing of unity is implied.
    connectivity : int
        The neighbourhood/connectivity considered when determining the surface
        of the binary objects. This value is passed to
        `scipy.ndimage.morphology.generate_binary_structure` and should usually be :math:`> 1`.
        Note that the connectivity influences the result in the case of the Hausdorff distance.

    Returns
    -------
    hd : float
        The symmetric Hausdorff Distance between the object(s) in ```result``` and the
        object(s) in ```reference```. The distance unit is the same as for the spacing of
        elements along each dimension, which is usually given in mm.

    See also
    --------
    :func:`hd`

    Notes
    -----
    This is a real metric. The binary images can therefore be supplied in any order.
    """
    hd1 = __surface_distances(result, reference, voxelspacing, connectivity)
    hd2 = __surface_distances(reference, result, voxelspacing, connectivity)
    hd95 = np.percentile(np.hstack((hd1, hd2)), 95)
    #
    hd95_mean = np.nanmean(hd95)
    return hd95_mean