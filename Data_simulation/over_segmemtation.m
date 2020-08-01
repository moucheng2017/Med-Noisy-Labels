clear;clc;
addpath('/media/le/Local Disk/BDMS/Scripts/NIfTI_20140122/');

Train_list = dir('/media/le/Local Disk/BDMS/MICCAI_BraTS_2019_Data_Training/HGG/');

for i = 3:3%length(Train_list)
    
    Subject_folder = strcat('/media/le/Local Disk/BDMS/MICCAI_BraTS_2019_Data_Training/HGG/', Train_list(i).name, '/');
    
    disp(strcat(num2str(i-2),'*******', Train_list(i).name));
    cd(Subject_folder);
    system('fslmaths BraTS19_2013_10_1_seg.nii.gz -edge mask_over_edge.nii.gz');
    system('fslmaths BraTS19_2013_10_1_seg.nii.gz -add mask_over_edge.nii.gz -bin mask_over_3');

%      for n =1:2
% 
%         system('fslmaths mask_over_3.nii.gz -edge mask_over_edge.nii.gz');
%         system('fslmaths mask_over_3.nii.gz -add mask_over_edge.nii.gz -bin mask_over_3');
%         n = n + 1;
%      end

    system('rm mask_over_edge.nii.gz');
end