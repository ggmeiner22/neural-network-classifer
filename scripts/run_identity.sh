#!/bin/bash
make || exit 1

# run both hidden configurations sequentially for comparison
./nn --mode identity --attr data/identity-attr.txt --train data/identity-train.txt --hidden 3 --lr 0.2 --momentum 0.9 --weight_decay 0 --epochs 3000000 --seed 1
./nn --mode identity --attr data/identity-attr.txt --train data/identity-train.txt --hidden 4 --lr 0.01 --momentum 0.7 --weight_decay 0 --epochs 100000 --seed 1