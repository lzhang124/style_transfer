#!/bin/bash

col='rgb-lin
    rgb-eig
    rgb-chol
    rgb-pca
    rgb-rot
    ycbcr-lin
    ycbcr-eig
    ycbcr-chol
    ycbcr-pca
    ycbcr-rot
    lab-lin
    lab-eig
    lab-chol
    lab-pca
    lab-rot'

python neural_style.py --content content/flowers_small.jpg --styles style/udnie.jpg --output output/flowers_udnie/flowers.jpg
python neural_style.py --content content/flowers_small.jpg --styles style/udnie.jpg --output output/flowers_udnie/flowers-luminance.jpg --luminance-transfer

for c in $col
do
   python neural_style.py --content content/flowers_small.jpg --styles style/udnie.jpg --output output/flowers_udnie/flowers-$c.jpg --map-colors $c
done
