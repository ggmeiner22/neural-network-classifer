#!/bin/bash
make | exit 1
./nn --mode tennis --attr data/tennis-attr.txt --train data/tennis-train.txt --test data/tennis-test.txt      --hidden 8 --lr 0.1 --momentum 0.9 --epochs 3