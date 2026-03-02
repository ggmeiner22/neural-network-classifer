#!/bin/bash
make | exit 1
./nn --mode iris_noisy --attr data/iris-attr.txt --train data/iris-train.txt --test data/iris-test.txt --hidden 8 --lr 0.05 --momentum 0.9 --epochs 2000 --valfrac 0.2 --seed 1