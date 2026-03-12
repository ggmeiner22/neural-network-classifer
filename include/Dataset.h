#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include "AttrParser.h"

// Dataset class handles loading data files and converting them
// into numerical feature vectors and label vectors for training
class Dataset {
public:
    // Constructor that stores the attribute file path and data file path
    Dataset(const std::string& attrFile,
            const std::string& dataFile);

    // Loads the dataset and converts it into feature and label matrices
    void load();

    // Returns the feature matrix X (inputs to the neural network)
    // Each element is a vector representing one training example
    const std::vector<std::vector<double>>& X() const;

    // Returns the label matrix Y (target outputs)
    // Labels may be one-hot encoded for classification tasks
    const std::vector<std::vector<double>>& Y() const;

    int inputSize() const;  // Returns number of input features per example
    int outputSize() const;  // Returns number of output values (number of classes or output bits)

private:
    AttrParser parser;  // Object responsible for reading the attribute schema file

    std::string dataFile;  // Path to the dataset file containing actual examples

    // Matrix of feature vectors
    // Each row represents one example's input values
    std::vector<std::vector<double>> features;

    // Matrix of label vectors
    // Each row represents the target output for an example
    std::vector<std::vector<double>> labels;
};

#endif