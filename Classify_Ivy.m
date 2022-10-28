% Term Project
% CSCI: 431 - Computer Vision
% Fall 2022
%
% ak3064 - Aditya Khanna
% arm5020 - Adam Mercer
% cju8676 - Corey Urbanke
function Classify_Ivy( filename )

    addpath('./IMAGES_of_POISON_IVY/')

    im = imread( filename );
    figure;
    imagesc( im );


end