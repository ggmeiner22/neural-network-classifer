#include "AttrParser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>

// --- helpers (C++11) ---
static std::string stripTrailingColon(const std::string& s) {
    if (!s.empty() && s[s.size() - 1] == ':') return s.substr(0, s.size() - 1);
    return s;
}

static bool startsWith(const std::string& s, const std::string& prefix) {
    return s.size() >= prefix.size() && s.compare(0, prefix.size(), prefix) == 0;
}

static bool isAllDigits(const std::string& s) {
    if (s.empty()) return false;
    for (size_t i = 0; i < s.size(); i++) {
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) return false;
    }
    return true;
}

static bool isOutToken(const std::string& name) {
    // out1, out2, ...
    if (!startsWith(name, "out")) return false;
    return isAllDigits(name.substr(3));
}

AttrParser::AttrParser(const std::string& filename)
    : filename(filename) {}

void AttrParser::parse() {
    attributes.clear();

    std::ifstream file(filename.c_str());
    if (!file) {
        std::cerr << "Error: cannot open attr file: " << filename << "\n";
        return;
    }

    std::string line;
    std::vector<Attribute> tmp;

    while (std::getline(file, line)) {
        // skip empty lines
        bool allSpace = true;
        for (size_t i = 0; i < line.size(); i++) {
            if (!std::isspace(static_cast<unsigned char>(line[i]))) { allSpace = false; break; }
        }
        if (allSpace) continue;

        std::istringstream iss(line);

        std::string rawName;
        if (!(iss >> rawName)) continue;

        std::string name = stripTrailingColon(rawName);

        // Gather remaining tokens
        std::vector<std::string> rest;
        std::string tok;
        while (iss >> tok) rest.push_back(tok);

        // If no rest tokens, skip
        if (rest.empty()) continue;

        Attribute attr;
        attr.name = name;
        attr.isNumeric = false;
        attr.categories.clear();

        // Iris numeric uses "continuous"
        // Example: SepalLength continuous
        if (rest.size() == 1 && (rest[0] == "continuous" || rest[0] == "numeric")) {
            attr.isNumeric = true;
        } else {
            // Tennis uses: Outlook Sunny Overcast Rain
            // Iris label uses: Iris Iris-setosa Iris-versicolor Iris-virginica
            // Identity uses: in1: 0 1  (categorical, but we can treat as categorical; Dataset will read as numeric anyway)
            attr.isNumeric = false;
            attr.categories = rest; // keep all categories as listed
        }

        tmp.push_back(attr);
    }

    // For identity: there is NO special "class" line; keep all attrs as schema, but Dataset will split in/out
    // For iris/tennis: last line is the class attribute (label categories)
    attributes = tmp;
}

const std::vector<Attribute>& AttrParser::getAttributes() const {
    return attributes;
}

int AttrParser::getClassIndex() const {
    // identity datasets do not have a final "class label" attribute line;
    // they instead have explicit out1..outN outputs in schema.
    // Detect identity by presence of out# attributes.
    int outCount = 0;
    for (size_t i = 0; i < attributes.size(); i++) {
        if (isOutToken(attributes[i].name)) outCount++;
    }
    if (outCount > 0) return -1;

    // otherwise (iris/tennis): last line is class
    if (attributes.empty()) return -1;
    return static_cast<int>(attributes.size()) - 1;
}