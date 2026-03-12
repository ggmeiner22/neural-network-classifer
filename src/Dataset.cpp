#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

// Checks whether string s begins with the given prefix
static bool startsWith(const std::string& s, const std::string& prefix) {
    // Make sure s is at least as long as prefix,
    // then compare the first prefix.size() characters of s to prefix
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

// Returns true only if the string contains digits only
static bool isAllDigits(const std::string& s) {
    // Empty strings are not considered all digits
    if (s.empty()) {
        return false;
    }
    // Loop through every character in the string
    for (size_t i = 0; i < s.size(); i++) {
        // If any character is not a digit, return false
        if (s[i] < '0' || s[i] > '9') {
            return false;
        }
    }
    return true;  // If every character was a digit, return true
}

static bool isInToken(const std::string& name) {
    // First confirm the name starts with "in"
    if (!startsWith(name, "in")) {
        return false;
    }
    // Then confirm everything after "in" is numeric
    return isAllDigits(name.substr(2));
}

static bool isOutToken(const std::string& name) {
    // First confirm the name starts with "out"
    if (!startsWith(name, "out")) {
        return false;
    }
    // Then confirm everything after "out" is numeric
    return isAllDigits(name.substr(3));
}

// Constructor
Dataset::Dataset(const std::string& attrFile,
                 const std::string& dataFile)
    : parser(attrFile), dataFile(dataFile) {}
// Initializes the attribute parser with the schema file
// and stores the dataset filename

void Dataset::load() {
    features.clear();  // Remove any old feature vectors
    labels.clear();  // Remove any old label vectors

    parser.parse();  // Parse the attribute schema file
    const std::vector<Attribute>& attrs = parser.getAttributes();  // Retrieve parsed attributes
    int classIndex = parser.getClassIndex();  // Determine index of class label attribute

    std::ifstream file(dataFile.c_str());  // Open the dataset file
    if (!file) {  // File failed to open
        std::cerr << "Error: cannot open data file: " << dataFile << "\n";
        return;
    }

    //  IDENTITY MODE (multi-output): in1..inK + out1..outM 
    if (classIndex == -1) {  // No single class attribute -> identity dataset
        int inCount = 0;
        int outCount = 0;

        // Count number of input and output attributes
        for (size_t i = 0; i < attrs.size(); i++) {
            if (isInToken(attrs[i].name)) {
                inCount++;
            } else if (isOutToken(attrs[i].name)) {
                outCount++;
            }
        }

        // Validate schema
        if (inCount == 0 || outCount == 0) {
            std::cerr << "Error: expected identity-style schema with in#/out# attributes.\n";
            return;
        }

        std::string line;
        // Read dataset by each line
        while (std::getline(file, line)) {
            // skip blank lines
            bool allSpace = true;
            for (size_t i = 0; i < line.size(); i++) {
                if (!std::isspace(static_cast<unsigned char>(line[i]))) { 
                    allSpace = false; 
                    break; 
                }
            }
            if (allSpace) {
                continue;
            }

            // Parse line tokens
            std::istringstream iss(line);
            std::vector<double> row;  // Stores all numeric values from line
            std::string tok;

            // Convert each token to a double + store in row
            while (iss >> tok) {
                row.push_back(std::atof(tok.c_str()));
            }

            // Ensure correct number of values
            if (static_cast<int>(row.size()) != inCount + outCount) {
                std::cerr << "Warning: identity row has " << row.size()
                          << " values; expected " << (inCount + outCount) << "\n";
                continue;
            }

            // Split row into input features and outputs
            std::vector<double> x(row.begin(), row.begin() + inCount);
            std::vector<double> y(row.begin() + inCount, row.end());

            features.push_back(x);  // Store input vector
            labels.push_back(y);  // Store output vector
        }

        return;
    }


    // CLASSIFICATION MODE (iris/tennis): last attr is label 
    if (attrs.empty() || classIndex < 0 || classIndex >= static_cast<int>(attrs.size())) {
        std::cerr << "Error: invalid attribute schema.\n";
        return;
    }

    // Retrieve class attribute
    const Attribute& classAttr = attrs[classIndex];

    // Ensure not empty
    if (classAttr.categories.empty()) {
        std::cerr << "Error: class attribute has no categories.\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // skip blank lines
        bool allSpace = true;
        for (size_t i = 0; i < line.size(); i++) {
            if (!std::isspace(static_cast<unsigned char>(line[i]))) { 
                allSpace = false; 
                break; 
            }
        }
        if (allSpace) {
            continue;
        }

        // Parse line tokens
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string tok;

        // Split line into tokens
        while (iss >> tok) {
            tokens.push_back(tok);
        }

        // Ensure correct number of values
        if (tokens.size() != attrs.size()) {
            std::cerr << "Warning: row token count (" << tokens.size()
                      << ") != attrs count (" << attrs.size() << ") in file " << dataFile << "\n";
            continue;
        }

        std::vector<double> x;  // Feature vector

        // build feature vector
        for (size_t i = 0; i < attrs.size(); i++) {
            if (static_cast<int>(i) == classIndex) {  // Skip label column
                continue;
            }

            const Attribute& a = attrs[i];
            const std::string& val = tokens[i];

            if (a.isNumeric) {  // Numeric feature
                x.push_back(std::atof(val.c_str()));
            } else {
                // If categorical -> convert to one-hot encoding

                // Ensure not empty
                if (a.categories.empty()) {
                    std::cerr << "Error: categorical attribute " << a.name << " has no categories.\n";
                    return;
                }
                bool found = false;
                for (size_t c = 0; c < a.categories.size(); c++) {
                    if (a.categories[c] == val) {
                        x.push_back(1.0);  // Matching category
                        found = true;
                    } else {
                        x.push_back(0.0);  // Not matching category
                    }
                }
                if (!found) {
                    std::cerr << "Warning: value '" << val << "' not found in categories for attribute "
                              << a.name << "\n";
                }
            }
        }

        // Build label vector (one-hot)
        std::vector<double> y(classAttr.categories.size(), 0.0);
        const std::string& labelTok = tokens[classIndex];

        int labelIndex = -1;

        // Find which category the label matches
        for (size_t k = 0; k < classAttr.categories.size(); k++) {
            if (classAttr.categories[k] == labelTok) {
                labelIndex = static_cast<int>(k);
                break;
            }
        }

        // Check if not no match or not found
        if (labelIndex == -1) {
            std::cerr << "Warning: label '" << labelTok << "' not found in class categories.\n";
            continue;
        }
        y[labelIndex] = 1.0;  // Set correct class index to 1

        features.push_back(x);  // Store feature vector
        labels.push_back(y);  // Store label vector
    }
}

// Returns reference to feature matrix
const std::vector<std::vector<double>>& Dataset::X() const {
    return features;
}

// Returns reference to label matrix
const std::vector<std::vector<double>>& Dataset::Y() const {
    return labels;
}

// Returns number of input features
int Dataset::inputSize() const {
    if (features.empty()) {
        return 0;
    }
    return static_cast<int>(features[0].size());
}

// Returns number of output features
int Dataset::outputSize() const {
    if (labels.empty()) {
        return 0;
    }
    return static_cast<int>(labels[0].size());
}