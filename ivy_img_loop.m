% function to loop through each input image and
% run classify Ivy on that image and assess accuracy
% of the classifier
function ivy_img_loop()

    addpath('./IMAGES_OTHER_PLANTS/');
    addpath('./IMAGES_of_POISON_IVY/');
    
    ivy_images = dir('IMAGES_of_POISON_IVY\*.jpg');
    for count = 1 : length( ivy_images )
        fname = ivy_images( count ).name;
        is_ivy = Classify_Ivy( fname );
        fprintf("%s - ", fname);
        if is_ivy
            fprintf("Poison IVY\n");
        else
            fprintf("NOT Poison IVY\n");
        end
    end
end