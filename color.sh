#!/bin/bash

declare -a col=("rgb-lin"
    "rgb-eig"
    "rgb-chol"
    "rgb-pca"
    "rgb-rot"
    "ycbcr-lin"
    "ycbcr-eig"
    "ycbcr-chol"
    "ycbcr-pca"
    "ycbcr-rot"
    "lab-lin"
    "lab-eig"
    "lab-chol"
    "lab-pca"
    "lab-rot")

python neural_style.py --content content/flowers_small.jpg --styles style/Robert_Delaunay,_1906,_Portrait.jpg --output output/flowers-luminance.jpg --luminance-transfer

for i in "${col[@]}"
do
   python neural_style.py --content content/flowers_small.jpg --styles style/Robert_Delaunay,_1906,_Portrait.jpg --output output/flowers-"-i".jpg --map-colors "-i"
done
