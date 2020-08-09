# Modelling Segmentation Annotators Pytorch
[![MIT License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE.md)

This repository contains our PyTorch implementation of our pre-print titled as ["Disentangling Human Error from the Ground Truth in Segmentation of Medical Images", 2020](http:). [Mou-Cheng Xu](https://moucheng2017.github.io/) is the main developer of the Python code; [Le Zhang](https://cheonglok.github.io/l.zhang/) is the main developer of the data simulation code.

<br>
 <img height="400" src="figures/humanerror_2.png" />
 </br>


# Contents
* [Introduction](#Introduction)
* [Installation](#Setup)
* [Download & preprocess the datasets](#Download)
* [Training](#Training)
* [Testing](#Testing)
* [Performance](#Performance)
* [Morphology Datasets](#Morphology)
* [How to cite this code](#MHow)

# Introduction

We present a method for jointly learning, from purely noisy observations alone, the reliability of individual annotators and the true segmentation label distributions, using two coupled CNNs. The separation of the two is achieved by encouraging the estimated annotators to be maximally unreliable while achieving high fidelity with the noisy training data.

**The architecture of our model is depicted below:**
<br>
 <img height="500" src="figures/NIPS.png" />
 </br>

# Installation
All required libraries can be installed via conda (anaconda). We recommend creating an conda env with all dependencies via environment file e.g., 
```sh
  conda env create -f conda_env.yml
```
# Download & preprocess the datasets

Download example datasets in following table as used in the paper, and pre-process the dataset using the folowing steps for multiclass segmentation purpose:

1. Download the training dataset with annotations from the corresponding link (e.g. Brats2019)
2. Unzip the data and you will have two folders: 
3. Extract the 2D images and annotations from nii.gz files by running
   
   ```sh
      - cd Brats
      - python ./preprocessing/Prepare_BRATS.py
   ```

| Dataset (with Link) | Content | Resolution (pixels) | Number of Classes |
| --- | --- | --- | --- |
| [MNIST](http://yann.lecun.com/exdb/mnist/)  | Handwritten Digits | 28 x 28 | 2 |
| [ISBI2015](https://smart-stats-tools.org/lesion-challenge) | Multiple Sclerosis Lesion  | 181 x 217 x 181 | 2 |
| [BraTS2019](https://www.med.upenn.edu/cbica/brats-2019/) | Multimodal Brain Tumor  | 240 x 240 x 155 | 4 |
| [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | Lung Image Database Consortium image collection | 180 x 180 | 2 |   

# Training
For BraTS dataset, set the hyper-parameters in `run.py`
   
   ```sh
      - input_dim=4,
      - class_no=4,
      - repeat=1,
      - train_batchsize=2,
      - validate_batchsize=1,
      - num_epochs=30,
      - learning_rate=1e-4,
      - alpha=1.5,
      - width=16,
      - depth=4,
      - data_path=your path,
      - dataset_tag='brats',
      - label_mode='multi',
      - save_probability_map=True,
      - low_rank_mode=False
   ```
and run: 

```
python run.py 
```

# Testing
To test our model, please run `segmentation.py` with the following setting:
 1. change the `model_path` to your pre-trained model;
 2. change the `test_path` to your testing data.

# Performance
1. Comparison of segmentation accuracy and error of CM estimation for different methods trained withdense labels (mean±standard deviation). The best results are shown in bald. Note that we count out the Oraclefrom the model ranking as it forms a theoretical upper-bound on the performance where true labels are known onthe training data.

| Models | Brats Dice (%) | Brats CM estimation | LIDC-IDRI Dice (%) | LIDC-IDRI CM estimation |
| --- | --- | --- | --- | --- |
| Naive CNN on mean labels | 29.42±0.58  |  n/a | 56.72±0.61  |  n/a  |
| Naive CNN on mode labels | 34.12±0.45  |  n/a | 58.64±0.47  |  n/a  |
| Probabilistic U-net  | 40.53±0.75   |  n/a  | 61.26±0.69  |  n/a   |
| STAPLE  | 46.73±0.17  | 0.2147±0.0103   | 69.34±0.58  | 0.0832±0.0043   | 
| Spatial STAPLE  | 47.31±0.21  | 0.1871±0.0094   | 70.92±0.18  |  0.0746±0.0057   |
| Ours without Trace | 49.03±0.34   | 0.1569±0.0072   | 71.25±0.12  | 0.0482±0.0038    |
| Ours | **53.47±0.24** |  **0.1185±0.0056**  | **74.12±0.19**|  **0.0451±0.0025** |
| Oracle (Ours but with known CMs)   | 67.13±0.14  | 0.0843±0.0029  | 79.41±0.17  | 0.0381±0.0021 |

2. Comparison of segmentation accuracy and error of CM estimation for different methods trained withonly one label available per image (mean±standard deviation). The best results are shown in bald.

| Models | Brats Dice (%) | Brats CM estimation | LIDC-IDRI Dice (%) | LIDC-IDRI CM estimation |	
| --- | --- | --- | --- | --- |  
| Naive CNN on mean & mode labels| 36.12±0.93  |  n/a | 48.36±0.79   |  n/a  |
| STAPLE  | 38.74±0.85  | 0.2956±0.1047  | 57.32±0.87  | 0.1715±0.0134     | 
| Spatial STAPLE   | 41.59±0.74  | 0.2543±0.0867  | 62.35±0.64  | 0.1419±0.0207    |
| Ours without Trace | 43.74±0.49   | 0.1825±0.0724   | 66.95±0.51  | 0.0921±0.0167   |
| Ours | **46.21±0.28**   | **0.1576±0.0487** | **68.12±0.48** | **0.0587±0.0098** |

3. The final segmentation of our model on BraTSand each annotator network predictions visualization.(Best viewed in colour: the target label is red.）

<br>
 <img height="500" src="figures/Brats_1.jpg" />
 </br>

4. Visualisation of segmentation labels on BraTS dataset: (a) GT and simulated annotator’ssegmentations (Annotator 1 - 5); (b) the predictions from the supervised models.)

<br>
 <img height="340" src="figures/brats-compare.jpg" />
 </br>

5. Visualisation of segmentation labels on LIDC-IDRI dataset: (a) GT and simulated annota-tor’s segmentations (Annotator 1 - 5); (b) the predictions from the supervised models.)

<br>
 <img height="340" src="figures/LIDC-compare.jpg" />
 </br>

# Morphology Datasets

We generate synthetic annotations from an assumed GT on MNIST, MS lesion and BraTS datasets, to generate efficacy of the approach in an idealised situation where the GT is known. We simulate a group of 5 annotators of disparate characteristics by performing morphological transformations (e.g., thinning, thickening, fractures, etc) on the ground-truth (GT) segmentation labels, using Morpho-MNIST software. In particular, the first annotator provides faithful segmentation (“good-segmentation”) with approximate GT, the second tends over-segment (“over-segmentation”), the third tends to under-segment (“under-segmentation”), the fourth is prone to the combination of small fractures and over-segmentation (“wrong-segmentation”) and the fifth always annotates everything as the background (“blank-segmentation”). To create synthetic noisy labels in multi-class scenario, we first choose a target class and then apply morphological operations on the provided GT mask to create 4 synthetic noisy labels at different patterns, namely, over-segmentation, under-segmentation, wrong segmentation and good segmentation. We create training data by deriving labels from the simulated annotators. Here we provide several example images in `data_simulation`.

<br>
 <img height="500" src="figures/Morph.png" />
 </br>


## Citation
If you use this code or the dataset for your research, please cite our paper:
```
@article{HumanError2020,
  title={Disentangling Human Error from the Ground Truth in Segmentation of Medical Images},
  author={Zhang, Le and Tanno, Ryutaro and Xu, Mou-Cheng and Jacob, Joseph and Ciccarelli, Olga and Barkhof, Frederik and C. Alexander, Daniel},
  journal={arXiv preprint arXiv:UPDATE},
  year={2020},
}
```

