#!/bin/bash
make || exit 1

# Run the program and capture output, then extract the accuracy table to a data file
TMP_RAW=iris_noisy_raw.txt
OUT_DAT=iris_noisy.dat

./nn --mode iris_noisy \
	--attr data/iris-attr.txt --train data/iris-train.txt --test data/iris-test.txt \
	--hidden 16 --lr 0.01 --momentum 0.9 --epochs 2000 --weight_decay 0.0001 --valfrac 0.2 --seed 8 \
	2>&1 | tee "$TMP_RAW"

# Extract the table printed after the header line 'Noise%  TestAcc(no-val)  TestAcc(with-val)'
# Format: noise  acc_no_val  acc_with_val
awk '/^Noise%/{p=1; next} p && NF>0{print $1, $(NF-1), $NF} p && NF==0{exit}' "$TMP_RAW" > "$OUT_DAT"

echo "wrote $OUT_DAT"

# Generate plot using the plotting helper
scripts/make_plot.sh "$OUT_DAT"