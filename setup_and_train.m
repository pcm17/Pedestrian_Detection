clc;
pos_im_dir = [ pwd '/person_detection_training_data/pos/' ];
png_names = dir(fullfile([pwd, '/person_detection_training_data/pos/*.png']));
jpg_names = dir(fullfile(pwd, '/person_detection_training_data/pos/*.jpg'));
pos_ims = [png_names; jpg_names];
num_pos_ims = size(pos_ims,1);

rng(0, 'twister');
neg_im_dir = [ pwd '/person_detection_training_data/neg/' ];
neg_crop_dir = [ pwd '/neg_crops/' ];
png_names = dir(fullfile([pwd, '/person_detection_training_data/neg/*.png']));
jpg_names = dir(fullfile(pwd, '/person_detection_training_data/neg/*.jpg'));
neg_ims = [png_names; jpg_names];

cropped_cols = 95;
cropped_rows = 159;

cell_size = 8;
num_crops = 0;
neg_hog = [];
pos_hog = [];
%% Lets generate enough negative crops to match the number of positive crops
while num_crops < num_pos_ims
    i = randi(size(neg_ims,1),1);
    im = imread([neg_im_dir neg_ims(i).name]);
    num_rows = size(im,1);
    num_cols = size(im,2);
    %imshow(im);
    %% Try to make 10 random crops per image
    for n = 1:10    
        x = randi(num_rows,1);
        y = randi(num_cols,1);
        if((y + cropped_cols <= num_cols) && (x + cropped_rows <= num_rows) && num_crops < num_pos_ims)
            cropped_im = im(x:x+cropped_rows,y:y+cropped_cols,:);
            %imshow(cropped_im);
            num_crops = num_crops + 1;
            % Extract HOG features
            h = vl_hog(im2single(cropped_im), cell_size);
            h = reshape(h,[1,(size(h,1)*size(h,2)*size(h,3))]);
            neg_hog = [neg_hog;h];
            %%% Save the cropped image
            imwrite(cropped_im, [neg_crop_dir 'crop_' num2str(n) neg_ims(i).name]); 
        end
    end
end
    
for i = 1:num_pos_ims
    im = imread([pos_im_dir pos_ims(i).name]);
    h = vl_hog(im2single(im), cell_size);
    h = reshape(h,[1,(size(h,1)*size(h,2)*size(h,3))]);
    pos_hog = [pos_hog;h];
end

neg_labels = zeros(num_crops,1);
pos_labels = ones(num_pos_ims,1);
Y_train = [neg_labels;pos_labels];
X_train = [neg_hog; pos_hog];

model = fitcecoc(X_train, Y_train);
    
    




