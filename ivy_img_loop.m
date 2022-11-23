% function to loop through each input image and
% run classify Ivy on that image and assess accuracy
% of the classifier
function ivy_img_loop()

    addpath('./IMAGES_OTHER_PLANTS/');
    addpath('./IMAGES_of_POISON_IVY/');
    

    % confusion matrix values
    % prediction is ivy when it is ivy
    true_pos = 0;
    % prediction is not ivy when it is ivy
    false_neg = 0;

    ivy_images = dir('IMAGES_of_POISON_IVY\*.jpg');
    for count = 1 : length( ivy_images )
        fname = ivy_images( count ).name;
        is_ivy = Classify_Ivy( fname );
        fprintf("%s - ", fname);
        if is_ivy
            fprintf("Poison IVY\n");
            true_pos = true_pos + 1;
        else
            fprintf("NOT Poison IVY\n");
            false_neg = false_neg + 1;
        end
    end

    fprintf("\n---- END IVY ---- BEGIN NOT IVY TRAINING ----\n");

    % prediction is ivy when it is not ivy
    false_pos = 0;
    % prediction is not ivy when it is not ivy
    true_neg = 0;
    
    not_ivy_images = dir('IMAGES_OTHER_PLANTS\*.jpg');
    for count = 1 : length( ivy_images )
        fname = not_ivy_images( count ).name;
        is_ivy = Classify_Ivy( fname );
        fprintf("%s - ", fname);
        if is_ivy
            fprintf("Poison IVY\n");
            false_pos = false_pos + 1;
        else
            fprintf("NOT Poison IVY\n");
            true_neg = true_neg + 1;
        end
    end
end