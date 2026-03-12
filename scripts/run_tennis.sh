#!/bin/bash
make | exit 1
./nn --mode tennis --attr data/tennis-attr.txt --train data/tennis-train.txt --test data/tennis-test.txt --hidden 5 --lr 0.05 --momentum 0.2 --epochs 50 --weight_decay 0.01
#--hidden 2 --lr 0.1 --momentum 0.7 --epochs 8 --weight_decay 0