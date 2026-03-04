# Neural Network Classifier - HW3 Implementation

## Compilation

From the Neural Network Classifier root directory, run:

    make

This will compile all source files in src/ and produce the executable nn.

## Running the Four Required Tests

### 1. testIdentity

Output accuracy and hidden/output values for identity dataset with 3 and 4 hidden units:

    ./scripts/run_identity.sh

This script runs:
- Identity with 3 hidden units: ./nn --mode identity --attr data/identity-attr.txt --train data/identity-train.txt --hidden 3 --lr 0.2 --momentum 0.9 --weight_decay 0 --epochs 3000000 --seed 1
- Identity with 4 hidden units: ./nn --mode identity --attr data/identity-attr.txt --train data/identity-train.txt --hidden 4 --lr 0.01 --momentum 0.7 --weight_decay 0 --epochs 100000 --seed 1

Output format: For each input sample, displays:
  - input bits
  - hidden values (2 decimals) with binary threshold at 0.5
  - output values (1 decimal)
  
Example: 10000000 -> 0.95 0.03 0.85 (1 0 1) -> 0.8  0.0  0.1  0.1  0.1  0.2  0.0  0.1

### 2. testTennis

Output training and test accuracy for tennis dataset:

    ./scripts/run_tennis.sh

This script runs: ./nn --mode tennis --attr data/tennis-attr.txt --train data/tennis-train.txt --test data/tennis-test.txt --hidden 4 --lr 0.05 --momentum 0.2 --epochs 50 --weight_decay 0.01

Output shows:
  - Train accuracy: % on training set
  - Test accuracy: % on test set

### 3. testIris

Output training and test accuracy for iris dataset:

    ./scripts/run_iris.sh

This script runs: ./nn --mode iris --attr data/iris-attr.txt --train data/iris-train.txt --test data/iris-test.txt --hidden 16 --lr 0.01 --momentum 0.9 --epochs 2000 --weight_decay 0.0001

Output shows:
  - Train accuracy: % on training set
  - Test accuracy: % on test set

### 4. testIrisNoisy

Output accuracy on uncorrupted test set for 0%-20% label corruption (2% increments), with and without validation-based early stopping:

    ./scripts/run_irisNoisy.sh

This script runs: ./nn --mode iris_noisy --attr data/iris-attr.txt --train data/iris-train.txt --test data/iris-test.txt --hidden 16 --lr 0.01 --momentum 0.9 --epochs 2000 --weight_decay 0.0001 --valfrac 0.2 --seed 8

Output is a table with columns:
  - Noise%: Percentage of training labels corrupted (0%, 2%, ..., 20%)
  - TestAcc(no-val): Test accuracy without using validation set
  - TestAcc(with-val): Test accuracy with validation-based early stopping

Also generates iris_noisy_compare.png plot comparing the two training approaches.

## Implementation Details

- Algorithm: Backpropagation with momentum and L2 weight decay
- Language: C++11
- Network: One hidden layer, configurable hidden units (3-16)
- Input Parameters Supported:
  - --hidden: Number of hidden units
  - --lr: Learning rate
  - --momentum: Momentum factor
  - --epochs: Number of training epochs (stopping criterion)
  - --weight_decay: L2 weight decay coefficient
  - --seed: Random seed for reproducibility
  - --valfrac: Validation split fraction (for early stopping)
  - --mode: One of identity, tennis, iris, iris_noisy

- Output Encoding: Discrete attributes converted to 1-of-n representation
- Loss Functions:
  - Classification (iris, tennis): Cross-entropy with softmax
  - Identity: Mean squared error with sigmoid

## Cleaning Build Artifacts

    make clean

This removes object files, executables, and generated data files.

## Prerequisites

Uses standard C++11 libraries only; no external dependencies.
Tested on Linux (Ubuntu 22.04), compiles with g++ with standard flags.
Requires bash for test scripts (scripts/run_*.sh).
