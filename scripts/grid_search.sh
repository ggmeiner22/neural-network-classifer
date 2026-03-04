#!/bin/bash
# Simple grid search for hyperparameters using the nn executable.
# Usage: ./scripts/grid_search.sh mode attr train [test]
#   mode: iris|tennis|iris_noisy|identity
#   test: omitted for identity (the script will copy train accuracy)
# Example:
#   ./scripts/grid_search.sh iris data/iris-attr.txt data/iris-train.txt data/iris-test.txt
#   ./scripts/grid_search.sh identity data/identity-attr.txt data/identity-train.txt
# The script loops over a predefined grid of hidden units, learning rates,
# momentums and weight decay values, runs the model, and appends results to
# results.txt in the current directory.

set -euo pipefail

if [ $# -lt 3 ]; then
    echo "Usage: $0 <mode> <attr> <train> [<test>]"
    exit 1
fi

MODE=$1
ATTR=$2
TRAIN=$3
TEST=""
if [ $# -ge 4 ]; then
    TEST=$4
fi

# specify grid here; edit as needed
# specify grid here; edit as needed
hiddens=(3 4)
lrs=(0.01 0.05 0.1 0.2)
moms=(0.0 0.1 0.2 0.5 0.7 0.8 0.9)
wds=(0 0.1 0.01 0.001 0.0001)
epochs=(100 500 1000 2500 5000 10000 50000)   # try multiple epoch counts

# compile first
make || exit 1

out_file="results_${MODE}.txt"
echo "hidden lr momentum weight_decay epochs train_acc test_acc" > "$out_file"

for h in "${hiddens[@]}"; do
    for lr in "${lrs[@]}"; do
        for m in "${moms[@]}"; do
            for wd in "${wds[@]}"; do
                for epo in "${epochs[@]}"; do
                    echo "Running h=$h lr=$lr mom=$m wd=$wd epochs=$epo"
                    # build command depending on mode
                    if [ "${MODE}" = "identity" ]; then
                        cmd=("./nn" "--mode" "$MODE" "--attr" "$ATTR" \
                             "--train" "$TRAIN" "--hidden" "$h" \
                             "--lr" "$lr" "--momentum" "$m" \
                             "--epochs" "$epo" "--weight_decay" "$wd" \
                             "--seed" "1")
                    else
                        cmd=("./nn" "--mode" "$MODE" "--attr" "$ATTR" \
                             "--train" "$TRAIN" "--test" "$TEST" \
                             "--hidden" "$h" "--lr" "$lr" \
                             "--momentum" "$m" "--epochs" "$epo" \
                             "--weight_decay" "$wd" \
                             "--seed" "1")
                    fi
                    out=$("${cmd[@]}" 2>&1)
                    # parse accuracy lines
                    if [ "${MODE}" = "identity" ]; then
                        # identity mode prints "Training exact-match accuracy (threshold 0.5): X"
                        train_acc=$(echo "$out" | grep "exact-match accuracy" | tail -1 | awk '{print $NF}')
                        test_acc=$train_acc
                    else
                        train_acc=$(echo "$out" | grep "Train accuracy" | tail -1 | awk '{print $3}')
                        test_acc=$(echo "$out" | grep "Test  accuracy" | tail -1 | awk '{print $3}')
                    fi
                    echo "$h $lr $m $wd $epo $train_acc $test_acc" >> "$out_file"
                done
            done
        done
    done

done

echo "Grid search complete, results in $out_file"
