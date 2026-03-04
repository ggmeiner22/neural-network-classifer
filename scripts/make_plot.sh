#!/bin/bash

# usage: make_plot.sh [datafile]
DATAFILE=${1:-iris_noisy.dat}
OUTPNG=${2:-iris_noisy_compare.png}

if [ ! -f "$DATAFILE" ]; then
     echo "data file '$DATAFILE' not found; expected columns: noise acc_no_val acc_with_val" >&2
     exit 1
fi

# create a cleaned temporary data file: strip CRs and keep only first 3 fields of numeric rows
CLEANFILE="${DATAFILE%.dat}.clean.dat"
tr -d '\r' < "$DATAFILE" | awk 'NF>=3 {print $1, $2, $3}' > "$CLEANFILE"

if [ ! -s "$CLEANFILE" ]; then
     echo "data file '$DATAFILE' contains no valid rows (expected 3 columns)" >&2
     exit 1
fi

     gnuplot -e "set terminal png size 600,400; set output '$OUTPNG'; set xlabel 'Noise %'; set ylabel 'Test accuracy'; set title 'Iris noisy: validation vs no validation'; set grid; set key left bottom; plot '$CLEANFILE' using 1:2 with linespoints lw 3 pt 7 ps 1.6 lc rgb 'blue' title 'no-val', '$CLEANFILE' using 1:3 with linespoints lw 1 pt 7 ps 1.4 lc rgb 'red' title 'with-val'"

echo "created $OUTPNG (from $CLEANFILE)"
