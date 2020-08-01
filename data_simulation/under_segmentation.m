clear;clc;

Train_list = dir('/media/le/Local Disk/BDMS/ISBI2015/Data/Testing_MS');

for i = 3:length(Train_list)
    
    Subject_folder = strcat('/media/le/Local Disk/BDMS/ISBI2015/Data/Testing_MS/', Train_list(i).name, '/');
    
    disp(strcat(num2str(i-2),'*******', Train_list(i).name));
    cd(Subject_folder);
    system('seg_maths mask.nii.gz -edge 0.9 mask_under_edge.nii.gz');
    system('fslmaths mask.nii.gz -sub mask_under_edge.nii.gz -bin mask_under_2');

    for n = 1:1

        system('seg_maths mask_under_2.nii.gz -edge 0.9 mask_under_edge.nii.gz');
        system('fslmaths mask_under_2.nii.gz -sub mask_under_edge.nii.gz -bin mask_under_2');
        n = n+1;
    end

    system('rm mask_under_edge.nii.gz');
end