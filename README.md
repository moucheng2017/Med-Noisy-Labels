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

<dl>
  <dd>(1) Comparison of segmentation accuracy and error of CM estimation for different methods trained withdense labels (mean±standard deviation). The best results are shown in bald. Note that we count out the Oraclefrom the model ranking as it forms a theoretical upper-bound on the performance where true labels are known onthe training data.

| Models | Brats Dice (%) | Brats CM estimation | LIDC-IDRI Dice (%) | LIDC-IDRI CM estimation |
| --- | --- | --- | --- | --- |
| Naive CNN on mean labels | 29.42 ± 0.58  |  n/a | 56.72 ± 0.61  |  n/a  |
| Naive CNN on mode labels | 34.12 ± 0.45  |  n/a | 58.64 ± 0.47  |  n/a  |
| Probabilistic U-net  | 40.53 ± 0.75   |  n/a  | 61.26 ± 0.69  |  n/a   |
| STAPLE  | 46.73 ± 0.17  | 0.2147 ± 0.0103   | 69.34 ± 0.58  | 0.0832 ± 0.0043   | 
| Spatial STAPLE  | 47.31 ± 0.21  | 0.1871 ± 0.0094   | 70.92 ± 0.18  |  0.0746 ± 0.0057   |
| Ours without Trace | 49.03 ± 0.34   | 0.1569 ± 0.0072   | 71.25 ± 0.12  | 0.0482 ± 0.0038    |
| Ours | **53.47 ± 0.24** |  **0.1185 ± 0.0056**  | **74.12 ± 0.19**|  **0.0451 ± 0.0025** |
| Oracle (Ours but with known CMs)   | 67.13 ± 0.14  | 0.0843 ± 0.0029  | 79.41 ± 0.17  | 0.0381 ± 0.0021 |

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
