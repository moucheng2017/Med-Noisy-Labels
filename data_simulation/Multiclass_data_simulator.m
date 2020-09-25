clear;clc;

addpath('E:\Google Drive\Works\UCL-MS\Scripts\NIfTI_20140122\');
folder_list = dir('C:\Users\Le\Desktop\Kevin\this_Training\Training\');

for i = 3:length(folder_list)

    Subject_folder = strcat('C:\Users\Le\Desktop\Kevin\this_Training\Training\', folder_list(i).name, '\');
    GT = load_untouch_nii(strcat(Subject_folder, 'multi-class.nii.gz'));
    disp(folder_list(i).name);
    
    [ROW, COL, DIM] = size(GT.img);
%     Target = GT;
%     Target.img = zeros(ROW, COL, DIM);
%     
%     for k = 1:DIM
%         for m = 1:ROW
%             for j = 1:COL
%                 if GT.img(m,j,k) == 1
%                    Target.img(m,j,k) = 1;  
%                 end
%             end
%         end
%     end
    
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
    
%     for k = 1:DIM
%         for m = 1:ROW
%             for j = 1:COL
%                 if Over.img(m,j,k) ~= 1
%                    Over.img(m,j,k) = GT.img(m,j,k);
%                 end
%             end
%         end
%     end
%     
%     for k = 1:DIM
%         for m = 1:ROW
%             for j = 1:COL
%                 if Under.img(m,j,k) ~= 1 && GT.img(m,j,k) ~= 1
%                    Under.img(m,j,k) = GT.img(m,j,k);
%                 elseif Under.img(m,j,k) ~= 1 && GT.img(m,j,k) == 1
%                     Under.img(m,j,k) = 4;
%                 end
%             end
%         end
%     end
    
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
    
%     for k = 1:DIM
%         for m = 1:ROW
%             for j = 1:COL
%                 if GT.img(m,j,k) ~= 0
%                    Blank.img(m,j,k) = 0;
%                 end
%             end
%         end
%         Blank.img(1,1,k) = 1;
%     end
    
    %save_untouch_nii(Good, strcat(Subject_folder, 'Good.nii.gz'));
    save_untouch_nii(Over, strcat(Subject_folder, 'multi-class-over.nii.gz'));
    save_untouch_nii(Under, strcat(Subject_folder, 'multi-class-under.nii.gz'));
    save_untouch_nii(Wrong, strcat(Subject_folder, 'multi-class-wrong.nii.gz'));
    %save_untouch_nii(Blank, strcat(Subject_folder, 'Blank.nii.gz'));
    
    %Gaussian.img(:,:,1) = double(Gaussian.img(:,:,1)) + randn(size(Gaussian.img(:,:,1)));
    Gaussian.img(:,:,2) = double(Gaussian.img(:,:,2)) + 0.2*randn(size(Gaussian.img(:,:,2)));
    %Gaussian.img(:,:,3) = double(Gaussian.img(:,:,3)) + randn(size(Gaussian.img(:,:,3)));
    w=fspecial('gaussian',[5 5],5);
    Blur.img=imfilter(Blur.img,w);
    
    save_untouch_nii(Gaussian, strcat(Subject_folder, 'multi-class-gaussian.nii.gz'));
    save_untouch_nii(Blur, strcat(Subject_folder, 'multi-class-blur.nii.gz'));
end