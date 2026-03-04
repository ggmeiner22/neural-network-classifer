#!/bin/bash
make | exit 1
./nn --mode iris_noisy --attr data/iris-attr.txt --train data/iris-train.txt --test data/iris-test.txt --hidden 16 --lr 0.01 --momentum 0.9 --epochs 2000 --weight_decay 0.0001 --valfrac 0.2 --seed 1