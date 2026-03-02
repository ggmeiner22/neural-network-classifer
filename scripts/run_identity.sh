#!/bin/bash
make
./nn --mode identity --attr data/identity-attr.txt --train data/identity-train.txt      --hidden 3 --lr 0.2 --momentum 0.9 --epochs 5000