clear;clc;

addpath('E:\Google Drive\Works\UCL-MS\Scripts\NIfTI_20140122\');
folder_list = dir('C:\Users\Le\Desktop\\this_Training\Training\');

for i = 3:length(folder_list)

    Subject_folder = strcat('C:\Users\Le\Desktop\this_Training\Training\', folder_list(i).name, '\');
    GT = load_untouch_nii(strcat(Subject_folder, 'multi-class.nii.gz'));
    disp(folder_list(i).name);
    
    [ROW, COL, DIM] = size(GT.img);
    
    se1 = strel('square', 3);
    Good = GT;
    Over = GT;
    Under = GT;
    Wrong = GT;
    Blank = GT;
    
    Gaussian = GT;
    Blur = GT;
    
    Close.img = imclose(GT.img, se1);
    Over.img = imdilate(GT.img, se1);
    Under.img = imerode(GT.img, se1);
    
    
    for k = 1:DIM
        for m = 1:ROW
            for j = 1:COL
                if GT.img(m,j,k) == 1
                   Wrong.img(m,j,k) = 4;
                elseif GT.img(m,j,k) == 4
                    Wrong.img(m,j,k) = 2;
                elseif GT.img(m,j,k) == 2
                    Wrong.img(m,j,k) = 1;
                end
            end
        end
    end
    
    
    save_untouch_nii(Good, strcat(Subject_folder, 'multi-class-good.nii.gz'));
    save_untouch_nii(Over, strcat(Subject_folder, 'multi-class-over.nii.gz'));
    save_untouch_nii(Under, strcat(Subject_folder, 'multi-class-under.nii.gz'));
    save_untouch_nii(Wrong, strcat(Subject_folder, 'multi-class-wrong.nii.gz'));
    save_untouch_nii(Blank, strcat(Subject_folder, 'multi-class-blank.nii.gz'));
    
end