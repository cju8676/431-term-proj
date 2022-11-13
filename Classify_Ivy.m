% Term Project
% CSCI: 431 - Computer Vision
% Fall 2022
function Classify_Ivy( filename )

    addpath('./IMAGES_OTHER_PLANTS/');
    addpath('./IMAGES_of_POISON_IVY/');

    % read in image
    im = ( imread( filename ) );

    % im -> remove noise -> feature extract
    function im_preprocessed = preprocess( im )
    
       % [row dimension, column dimension]
        dims = size(im);
        % the pixel in the very center of the image [x, y]
        center = round( dims( 1:2 ) / 2 );
    
        % info of our circle [row location, col location, circle radius]
        ci = [center(1), center(2), center(1)]; 
        % grid wrt our circle location
        [xx,yy] = ndgrid( ( 1:dims(1) ) - ci(1),( 1:dims(2)) - ci(2) );
        % get the pixels inside the circle radius ci(3)
        mask = uint8((xx.^2 + yy.^2) < ci(3)^2);
        im_leaf_center = uint8(zeros(size(im)));
        % mask each of our rgb channels
        im_leaf_center(:,:,1) = im(:,:,1).*mask;
        im_leaf_center(:,:,2) = im(:,:,2).*mask;
        im_leaf_center(:,:,3) = im(:,:,3).*mask;
    
        % kmeans takes way too long on the full resolution image
        im_smaller = im_leaf_center( 3:3:end, 3:3:end, : );
        s_dims = size(im_smaller);
        % row vectors, column vectors of our image - tells us where a pixel is
        [xs, ys] = meshgrid( 1:s_dims(1), 1:s_dims(2) );
    
        figure;
        imagesc(im_smaller);
        im_smaller_hsv = rgb2hsv(im_smaller);
        % 75% more saturation:
        im_smaller_hsv(:, :, 2) = im_smaller_hsv(:, :, 2) * 1.75;
        im_smaller_sat = hsv2rgb(im_smaller_hsv);
        figure;
        imagesc(im_smaller_sat);

        % Convert our image to CIELAB 
        im_lab = rgb2lab(im_smaller_sat);
    
        % Store luminance, a*, and b* values to be used for k-means
        lum = im_lab(:,:,1);
        a_star = im_lab(:,:,2);
        b_star = im_lab(:,:,3);
        
        % Attributes input to k-means, space variables weighted by 1/15
        attributes = [xs(:)/10, ys(:)/10, lum(:), a_star(:), b_star(:)];
        % Timer Start
        tic;
        % Perform kmeans clustering on image
        [cluster_id, centers] = kmeans(attributes, 15, 'MaxIter',250);
        % Timer Stop
        toc;
    
        % Show our resulting clustered image by reshaping the clusters we got
        % from k-means
        figure;
        im_new = reshape(cluster_id, s_dims(1), s_dims(2));
        imagesc(im_new);
        centers_colors = lab2rgb(centers(:,3:5));
%         colormap(centers_colors);
        title("Clustered Image - k = 5, wt = 1/15")
        colorbar;
        drawnow;
        axis image;

        new_cen = [centers(:, 1)*10 centers(:, 2)*10 centers(:, 3) centers(:, 4) centers(:, 5)];
        [green_rows, ~] = find(new_cen(:, 4) < -10);
        green_cen = [];
        for i = 1 : size(green_rows)
            green_cen = [green_cen ; new_cen(green_rows(i), :)];
        end

%         for i = 1 : size(green_cen)
%             d = pdist2([center ; green_cen(i, 1:2)]);
%             if d < smallest_dist
%                 smallest_dist = d;
%                 idx_smallest = i;
%             end
%         end
%         d = pdist2(center, green_cen(1:2));
        % which green cluster has the most pixels on target
        green_cluster = 0;
        greatest_num_green_pix = 0;
        for i = 1 : size(green_cen)
        
            [~, cluster] = ismember(green_cen(i, :), new_cen, 'rows');
            im_green_new = (im_new == cluster);
            green_sum = sum(im_green_new(:));
            if green_sum > greatest_num_green_pix
                greatest_num_green_pix = green_sum;
                green_cluster = cluster;
            end
        
        end
        % find green_cen of index and find green_cen in new_cen
        % this is the cluster we want
% %         smallest_green_cen = green_cen(idx_smallest, :);
% %         green = find(ismember(smallest_green_cen, new_cen, 'rows'));
    
        % for now get the cluster with the highest green value
        % this == lowest a* valued cluster
        % centers colors is an array of clusters with rows
        % being [ x y lum a* b* ]
%         most_green = min(centers(:, 4));
%         [most_row, ~] = find(centers == most_green);
        im_new_leaf = (im_new == green_cluster);
%         [rgb_max, rgb_idx] = max(centers_colors);
%         im_new_leaf = (im_new == rgb_idx(2));
        
        % Disk structuring element
        disk = strel('disk', 5);
        % Get rid of black specs on leaf
        im_dilate_leaf = imdilate(im_new_leaf, disk);
        % Make sure objects picked up are separated enough
        im_final_morph = imerode(im_dilate_leaf, disk);
        % Label and get number of our different blobs - 4 pixel connectivity
        [L, n] = bwlabel(im_final_morph, 4);
        display(n);
        
        im_final_preprocessed = zeros(size(im_final_morph));
        % loop through discovered blobs
        for d = 1 : n
            
            this_blob = (L == d);
            
            stats = regionprops(this_blob, 'all');
%             display(stats);
            if stats.Area < 10000
                continue;
            end
            display(stats);
            % BoundingBox = [left, top, width height]
            blob_wid = stats.BoundingBox(3);
            blob_hei = stats.BoundingBox(4);
            % if we are close to a 1:1 ratio, its a square blob, and most
            % likely a die blob
            % to be safe lets try a range of .75 to 1.25
%             ratio = blob_wid / blob_hei;
%             if ratio < .75 || ratio > 1.25
%                 continue;
%             end
%             corners = detectHarrisFeatures(this_blob, 'MinQuality', 0.5);
%             figure;
%             display(corners);
            im_final_preprocessed = im_final_preprocessed | this_blob;
            imagesc(this_blob);
            hold on;
%             plot(corners.selectStrongest(50));
            pause(2);
        end
        im_preprocessed = im_final_preprocessed;
    end
    
    % fit model -> return our classification of ivy or not ivy
    function is_ivy = classify( feat )
        
        % if leaf != 3 leaves, !ivy

        % if each leaf has only 1-2 thumbs (corners), ivy

        % if !green, !ivy

    end

    % pre process leaf image
    im_pre = preprocess(im);
    
    
    
    figure;
    imagesc(im_pre);
    colormap("gray");

end