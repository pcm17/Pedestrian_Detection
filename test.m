png_names = dir(fullfile([pwd, '/test_images/*.png']));
jpg_names = dir(fullfile(pwd, '/test_images/*.jpg'));
test_im_dir = [ pwd '/test_images/' ];
person_ims_dir = [pwd '/found_persons/' ];
test_ims = [png_names; jpg_names];
num_test_ims = size(test_ims,1);

for i = 1:num_test_ims
    im = imresize(im2single(imread([test_im_dir test_ims(i).name])), 0.95);
    window_width = 95;
    window_height = 159;
    image_width = size(im,2);
    image_height = size(im,1);
    
    y = 1;  x = 1;  n = 1;
    while (y + window_height <= image_height)
        if x + window_width >= image_width
            x = 1;
            y = y + 50;
        end
        if y + window_height <=  image_height && x + window_width <= image_width
           window = im(y:y+window_height,x:x+window_width,:);
           h = vl_hog(window, cell_size);
           h = reshape(h,[1,(size(h,1)*size(h,2)*size(h,3))]);
           %%% Classify image as containing a person or not
           label = predict(model, h);
           if label == 1
               imshow(window);
               imwrite(window, [person_ims_dir test_ims(i).name num2str(n) '.png']);
               n = n + 1;
               %break;
           end
        end
        x = x + 5;  % Jump 5 rows down to begin next pass over image
    end
end