# LearnNoisyLabels_Pytorch
**The architecture of our model is depicted below:**
<br>
 <img height="500" src="images/NIPS_1.png" />
 </br>

**Below see testing samples from Brats and LIDC data set:**
<br>
 <img height="500" src="images/Brats_1.jpg" />
 </br>

<br>
 <img height="500" src="images/LIDC.jpg" />
 </br>


# Setup package in virtual environment
```sh
  - git clone https://github.com/UCLBrain/LearnNoisyLabels_Pytorch
  - cd LearnNoisyLabels_Pytorch/
  - conda env create -f conda_env.yml
```
# Download & preprocess the Brats dataset
<dl>
  <dd>(1) Download the training dataset with annotations from: https://www.med.upenn.edu/cbica/brats2019/registration.html
  <dd>(2) Unzip the data and you will have two folders: 
  <dd>(3) Extract the 2D images from nii.gz files by running
   ```sh
    - cd Brats
    - python Prepare_BRATS.py
   ```
