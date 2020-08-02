# Modelling_Segmentation_Annotators_Pytorch

# Table of Contents
* [Introduction](#Introduction)
* [Setup package in virtual environment](#Setup)
* [Download & preprocess the datasets](#Download)
* [Training](#Training)
* [Evaluation](#Evaluation)
* [Performance](#Performance)
* [Morphology Datasets](#Morphology)
* [How to cite this code](#MHow)

# Introduction

We present a method for jointly learning, from purely noisy observations alone, the reliability of individual annotators and the true segmentation label distributions, using two coupled CNNs. The separation of the two is achieved by encouraging the estimated annotators to be maximally unreliable while achieving high fidelity with the noisy training data.

**The architecture of our model is depicted below:**
<br>
 <img height="500" src="figures/NIPS.png" />
 </br>

# Setup package in virtual environment
```sh
  - git clone https://github.com/UCLBrain/LearnNoisyLabels_Pytorch
  - cd LearnNoisyLabels_Pytorch/
  - conda env create -f conda_env.yml
```
# Download & preprocess the datasets

Download example datasets in following table as used in the paper, and pre-process the dataset using the folowing steps for multiclass segmentation purpose:

<dl>
  <dd>(1) Download the training dataset with annotations from the corresponding link (e.g. Brats2019)
  <dd>(2) Unzip the data and you will have two folders: 
  <dd>(3) Extract the 2D images and annotations from nii.gz files by running
   
   ```sh
      - cd Brats
      - python Prepare_BRATS.py
   ```

| Dataset (with Link) | Content | Resolution (pixels) | Number of Classes |
| --- | --- | --- | --- |
| [MNIST](http://yann.lecun.com/exdb/mnist/)  | Handwritten Digits | 28 x 28 | 2 |
| [ISBI2015](https://smart-stats-tools.org/lesion-challenge) | Multiple Sclerosis Lesion  | 181 x 217 x 181 | 2 |
| [Brats2019](https://www.med.upenn.edu/cbica/brats-2019/) | Multimodal Brain Tumor  | 181 x 217 x 181 | 4 |
| [LIDC-IDRI](https://wiki.cancerimagingarchive.net/display/Public/LIDC-IDRI) | Lung Image Database Consortium image collection | 181 x 217 x 181 | 2 |

# Training
<dl>
  <dd>(1) For Brats dataset, set the parameters in Test.py
   
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
      - low_rank_mode=False,
      - rank=0,
      - epoch_threshold=0,
      - alpha_initial=-1.5,
      - regularisation_type='2'
   ```
   <dd>(2) source activate env
   <dd>(3) python NNTrain.py
    
# Evaluation

# Performance

**Below see testing samples from Brats dataset:**
<br>
 <img height="500" src="figures/Brats_1.jpg" />
 </br>

<br>
 <img height="350" src="figures/brats-compare.jpg" />
 </br>

# Morphology Datasets

<br>
 <img height="500" src="figures/Morph.png" />
 </br>

# How to cite this code
Please cite the original publication:
