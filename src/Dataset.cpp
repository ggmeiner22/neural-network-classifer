#include "Dataset.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstdlib>

// --- helpers ---
static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool isAllDigits(const std::string& s) {
    if (s.empty()) return false;
    for (size_t i = 0; i < s.size(); i++) {
        if (s[i] < '0' || s[i] > '9') return false;
    }
    return true;
}

static bool isInToken(const std::string& name) {
    if (!startsWith(name, "in")) return false;
    return isAllDigits(name.substr(2));
}

static bool isOutToken(const std::string& name) {
    if (!startsWith(name, "out")) return false;
    return isAllDigits(name.substr(3));
}

Dataset::Dataset(const std::string& attrFile,
                 const std::string& dataFile)
    : parser(attrFile), dataFile(dataFile) {}

void Dataset::load() {
    features.clear();
    labels.clear();

    parser.parse();
    const std::vector<Attribute>& attrs = parser.getAttributes();
    int classIndex = parser.getClassIndex();

    std::ifstream file(dataFile.c_str());
    if (!file) {
        std::cerr << "Error: cannot open data file: " << dataFile << "\n";
        return;
    }

    // --- IDENTITY MODE (multi-output): in1..inK + out1..outM ---
    if (classIndex == -1) {
        int inCount = 0;
        int outCount = 0;
        for (size_t i = 0; i < attrs.size(); i++) {
            if (isInToken(attrs[i].name)) inCount++;
            else if (isOutToken(attrs[i].name)) outCount++;
        }

        if (inCount == 0 || outCount == 0) {
            std::cerr << "Error: expected identity-style schema with in#/out# attributes.\n";
            return;
        }

        std::string line;
        while (std::getline(file, line)) {
            // skip empty
            bool allSpace = true;
            for (size_t i = 0; i < line.size(); i++) {
                if (!std::isspace(static_cast<unsigned char>(line[i]))) { allSpace = false; break; }
            }
            if (allSpace) continue;

            std::istringstream iss(line);
            std::vector<double> row;
            std::string tok;

            while (iss >> tok) {
                // identity data is numeric tokens (0/1)
                row.push_back(std::atof(tok.c_str()));
            }

            if (static_cast<int>(row.size()) != inCount + outCount) {
                std::cerr << "Warning: identity row has " << row.size()
                          << " values; expected " << (inCount + outCount) << "\n";
                continue;
            }

            std::vector<double> x(row.begin(), row.begin() + inCount);
            std::vector<double> y(row.begin() + inCount, row.end());

            features.push_back(x);
            labels.push_back(y);
        }

        return;
    }

    // --- CLASSIFICATION MODE (iris/tennis): last attr is label ---
    if (attrs.empty() || classIndex < 0 || classIndex >= static_cast<int>(attrs.size())) {
        std::cerr << "Error: invalid attribute schema.\n";
        return;
    }

    const Attribute& classAttr = attrs[classIndex];
    if (classAttr.categories.empty()) {
        std::cerr << "Error: class attribute has no categories.\n";
        return;
    }

    std::string line;
    while (std::getline(file, line)) {
        // skip empty
        bool allSpace = true;
        for (size_t i = 0; i < line.size(); i++) {
            if (!std::isspace(static_cast<unsigned char>(line[i]))) { allSpace = false; break; }
        }
        if (allSpace) continue;

        std::istringstream iss(line);

        std::vector<std::string> tokens;
        std::string tok;
        while (iss >> tok) tokens.push_back(tok);

        if (tokens.size() != attrs.size()) {
            std::cerr << "Warning: row token count (" << tokens.size()
                      << ") != attrs count (" << attrs.size() << ") in file " << dataFile << "\n";
            continue;
        }

        std::vector<double> x;

        // build features
        for (size_t i = 0; i < attrs.size(); i++) {
            if (static_cast<int>(i) == classIndex) continue;

            const Attribute& a = attrs[i];
            const std::string& val = tokens[i];

            if (a.isNumeric) {
                x.push_back(std::atof(val.c_str()));
            } else {
                // one-hot categorical
                if (a.categories.empty()) {
                    std::cerr << "Error: categorical attribute " << a.name << " has no categories.\n";
                    return;
                }
                bool found = false;
                for (size_t c = 0; c < a.categories.size(); c++) {
                    if (a.categories[c] == val) {
                        x.push_back(1.0);
                        found = true;
                    } else {
                        x.push_back(0.0);
                    }
                }
                if (!found) {
                    std::cerr << "Warning: value '" << val << "' not found in categories for attribute "
                              << a.name << "\n";
                }
            }
        }

        // build label one-hot
        std::vector<double> y(classAttr.categories.size(), 0.0);
        const std::string& labelTok = tokens[classIndex];

        int labelIndex = -1;
        for (size_t k = 0; k < classAttr.categories.size(); k++) {
            if (classAttr.categories[k] == labelTok) {
                labelIndex = static_cast<int>(k);
                break;
            }
        }
        if (labelIndex == -1) {
            std::cerr << "Warning: label '" << labelTok << "' not found in class categories.\n";
            continue;
        }
        y[labelIndex] = 1.0;

        features.push_back(x);
        labels.push_back(y);
    }
}

const std::vector<std::vector<double>>& Dataset::X() const {
    return features;
}

const std::vector<std::vector<double>>& Dataset::Y() const {
    return labels;
}

int Dataset::inputSize() const {
    if (features.empty()) return 0;
    return static_cast<int>(features[0].size());
}

int Dataset::outputSize() const {
    if (labels.empty()) return 0;
    return static_cast<int>(labels[0].size());
}