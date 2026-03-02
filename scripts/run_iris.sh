#!/bin/bash
make
./nn --mode iris --attr data/iris-attr.txt --train data/iris-train.txt --test data/iris-test.txt      --hidden 8 --lr 0.05 --momentum 0.9 --epochs 2000