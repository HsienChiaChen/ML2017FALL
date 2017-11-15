#!/bin/sh

wget http://github.com/HsienChiaChen/ML2017FALL/releases/download/hw3_model/model-0.62792.h5
wget http://github.com/HsienChiaChen/ML2017FALL/releases/download/hw3_model/model-0.62898.h5
wget http://github.com/HsienChiaChen/ML2017FALL/releases/download/hw3_model/model-0.63715.h5
wget http://github.com/HsienChiaChen/ML2017FALL/releases/download/hw3_model/model-0.63889.h5
python3 mytest.py $1 $2 model-0.62792.h5 model-0.62898.h5 model-0.63715.h5 model-0.63889.h5 
