#ifndef DATASET_H
#define DATASET_H

#include <vector>
#include <string>
#include "AttrParser.h"

class Dataset {
public:
    Dataset(const std::string& attrFile,
            const std::string& dataFile);

    void load();

    const std::vector<std::vector<double>>& X() const;
    const std::vector<std::vector<double>>& Y() const;

    int inputSize() const;
    int outputSize() const;

private:
    AttrParser parser;

    std::string dataFile;
    std::vector<std::vector<double>> features;
    std::vector<std::vector<double>> labels;
};

#endif