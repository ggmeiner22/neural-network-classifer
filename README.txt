neural-network-classifer
This project implements a multi-layer perceptron neural network for classification and identity mapping tasks. The system loads datasets from attribute and data files, converts them into numerical feature vectors, trains the neural network using backpropagation, and evaluates model performance using accuracy metrics. Experiments include classification on the Tennis and Iris datasets as well as robustness testing with noisy labels.

Compalation and Execution
chmod -R u+w .
chmod +x *.sh
chmod +x scripts/*.sh
chmod +x src/*.cpp

Run All
./run_all.sh

Run Individually
Build
make

Identity
./scripts/run_identity.sh

Tennis
./scripts/run_tennis.sh

Iris
./scripts/run_iris.sh

IrisNoisy
./scripts/run_irisNoisy.sh

Clean object files
make clean


File Structure
neural-network-classifer/
├── data/
│   ├── identity-attr.txt
│   ├── identity-train.txt
│   ├── iris-attr.txt
│   ├── iris-test.txt
│   ├── iris-train.txt
│   ├── tennis-attr.txt
│   ├── tennis-test.txt
│   └── tennis-train.txt
├── include/
│   ├── AttrParser.h
│   ├── Dataset.h
│   ├── MLP.h
│   └── Util.h
├── scripts/
│   ├── run_identity.sh
│   ├── run_iris.sh
│   ├── run_irisNoisy.sh
│   └── run_tennis.sh
├── src/
│   ├── AttrParser.cpp
│   ├── Dataset.cpp
│   ├── MLP.cpp
│   ├── Util.cpp
│   └── main.cpp
|
├── Makefile
└── run_all.sh

## File Overview

### AttrParser.h / AttrParser.cpp
Handles parsing of attribute schema files that describe the dataset structure.
- **Attribute** structure stores metadata for each feature including the name, whether it is numeric, and possible categorical values.
- **AttrParser** reads attribute files and constructs a list of attribute definitions.
 -Detects numeric attributes (`continuous` / `numeric`) and categorical attributes with explicit value lists.
- Determines the class attribute index for classification datasets.
- Supports identity-style datasets by detecting `out#` attributes instead of a single class label.
> This module reads the schema and organizes the attribute data so other parts of the system can use it.

### Dataset.h / Dataset.cpp
Responsible for loading dataset files and converting them into numerical matrices used by the neural network.
- Uses `AttrParser` to interpret the dataset schema.
- Reads raw data files and converts them into:
- **Feature matrix **
- **Label matrix Y**
- Handles both classification datasets and identity mapping datasets.
- Automatically converts categorical attributes into one-hot encoded feature vectors.
- Splits identity datasets into `in#` input features and `out#` output targets.
> This module handles loading and preparing the data so the neural network only works with numeric values.

### MLP.h / MLP.cpp
Implements the Multi-Layer Perceptron (MLP) neural network used for learning.
- Weight initialization for input -> hidden and hidden -> output layers
- Forward propagation through the network
- Hidden layer sigmoid activations
- Output layer activation:
  - **Softmax** for classification tasks
  - **Sigmoid** for identity mapping tasks
- Backpropagation for computing gradients
- Stochastic gradient descent training
- Momentum
- Weight Decay

### Util.h / Util.cpp
Provides reusable utility functions used throughout the experiments.
- `argmax()` – returns the index of the maximum value in a vector
- `accuracyClass()` – computes classification accuracy
- `accuracyIdentityExact()` – computes exact-match accuracy for identity tasks
- `splitTrainVal()` – deterministic training/validation dataset split
- `corruptOneHotLabels()` – introduces controlled label noise for experiments
> This module abstracts evaluation and experiment utilities from the neural network implementation.

### main.cpp
Responsibilities include:
- Parsing command-line arguments
- Configuring experiment parameters
- Selecting which experiment mode to run
- Loading datasets
- Creating and training the neural network
- Reporting experiment results  

Supported experiment modes:   
- **identity** – trains the network to reproduce input bit patterns
- **tennis** – classification experiment using the PlayTennis dataset
- **iris** – classification experiment using the Iris dataset
- **iris_noisy** – evaluates robustness to label noise with and without validation
> Also controls random seeding to ensure reproducible experiments.

### data/
Contains datasets used for experiments.
- **identity**-* – binary identity mapping dataset
- **tennis**-* – discrete attribute classification dataset
- **iris**-* – continuous attribute classification dataset
> Each dataset includes an attribute file describing the schema and corresponding training/testing files.

### scripts/
Automation scripts for running individual experiments.
- `run_identity.sh`
- `run_tennis.sh`
- `run_iris.sh`
- `run_irisNoisy.sh`

Makefile
Defines compilation rules for building the project.
- Uses C++11 standard
- Compiles source files into object files
- Links the final executable
> Ensures consistent builds across machines

run_all.sh
- Builds the project
- Runs all experiment modes
> Provides a reproducible workflow for testing the neural network on all datasets.


