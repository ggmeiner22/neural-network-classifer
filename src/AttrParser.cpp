#include "AttrParser.h"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cctype>

// Removes a trailing colon from a string if one exists
// Used for identity-attr.txt
static std::string stripTrailingColon(const std::string& s) {
    // Ensure that the string is not empty and that its last character is ':'
    if (!s.empty() && s[s.size() - 1] == ':') {
        return s.substr(0, s.size() - 1);
    }
    // Otherwise return the original string unchanged
    return s;
}

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
        if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
            return false;
        }
    }
    return true;  // If every character was a digit, return true
}

// Checks whether a name matches the pattern out1, out2, out3, ...
static bool isOutToken(const std::string& name) {
    // First confirm the name starts with "out"
    if (!startsWith(name, "out")) {
        return false;
    }
    // Then confirm everything after "out" is numeric
    return isAllDigits(name.substr(3));
}

// Constructor for AttrParser that stores the filename to be parsed
AttrParser::AttrParser(const std::string& filename)
    : filename(filename) {}  // Initialize the member variable filename using an initializer list

// Reads the attribute schema file
void AttrParser::parse() {
    attributes.clear();  // Remove any previously stored attributes before parsing again

    std::ifstream file(filename.c_str());  // Open the input file using the stored filename
    if (!file) {  // File failed to open
        std::cerr << "Error: cannot open attr file: " << filename << "\n";
        return;
    }

    std::string line;  // Will hold each full line read from the file
    std::vector<Attribute> tmp; // Temporary list to collect parsed attributes

    // Read the file one line at a time
    while (std::getline(file, line)) {
        // Assume the line is all whitespace until proven otherwise
        bool allSpace = true;  

        // Check every character in the line
        for (size_t i = 0; i < line.size(); i++) {
            // If a non-whitespace character is found, mark the line as not empty
            if (!std::isspace(static_cast<unsigned char>(line[i]))) { 
                allSpace = false; 
                break; 
            }
        }
        // skip empty lines
        if (allSpace) {
            continue;
        }

        // Create a string stream so the line can be parsed token by token
        std::istringstream iss(line);

        std::string rawName;  // Stores the first token on the line, which should be the attribute name
        // If no token could be read, skip this line
        if (!(iss >> rawName)) {
            continue;
        }

        // Remove a trailing colon from the attribute name if one is present
        std::string name = stripTrailingColon(rawName);

        // Gather remaining tokens
        std::vector<std::string> rest;  // Will store the remaining tokens after the attribute name
        std::string tok;  // Temporary variable for reading each token
        // Read the rest of the tokens on the line
        while (iss >> tok) {
            rest.push_back(tok);
        }

        // If no rest tokens, skip
        if (rest.empty()) {
            continue;
        }

        Attribute attr;  // Create an Attribute object for this line
        attr.name = name;  // Store the cleaned attribute name
        attr.isNumeric = false;  // Default assumption -> attribute is categorical
        attr.categories.clear();  // Ensure the categories list starts empty

        // Iris numeric uses -> "continuous"
        // Example: SepalLength continuous
        if (rest.size() == 1 && (rest[0] == "continuous" || rest[0] == "numeric")) {
            attr.isNumeric = true;
        } else {
            // Tennis uses -> Outlook Sunny Overcast Rain
            // Iris label uses -> Iris Iris-setosa Iris-versicolor Iris-virginica
            // Identity uses -> in1: 0 1  
            // // These are categorical attributes, so store all listed category values
            attr.isNumeric = false;
            attr.categories = rest;  // Save the remaining tokens as the list of valid categories
        }
        // Add the parsed attribute to the temporary list
        tmp.push_back(attr);
    }

    // For identity -> there is NO special "class" line so keep all attrs as schema, but Dataset will split in/out
    // For iris/tennis: last line is the class attribute (label categories)
    attributes = tmp;
}

// Returns the parsed attribute list
const std::vector<Attribute>& AttrParser::getAttributes() const {
    return attributes;
}

// Determines which attribute index is the class label
int AttrParser::getClassIndex() const {
    // identity datasets do not have a final "class label" attribute line;
    // they instead have explicit out1..outN outputs in schema.
    // Detect identity by presence of out# attributes.
    int outCount = 0;

    // Loop through all parsed attributes
    for (size_t i = 0; i < attributes.size(); i++) {
        // If this attribute name matches the out# pattern, increment the count
        if (isOutToken(attributes[i].name)) {
            outCount++;
        }
    }
    // If output attributes exist, there is no single class index
    if (outCount > 0) {
        return -1;
    }

    // otherwise (iris/tennis): last line is class
    // If no attributes exist, no valid class index can be returned
    if (attributes.empty()) {
        return -1;
    }
    
    // Otherwise return the last attribute as the class label
    return static_cast<int>(attributes.size()) - 1;
}