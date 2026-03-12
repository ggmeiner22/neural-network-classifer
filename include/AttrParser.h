#ifndef ATTR_PARSER_H
#define ATTR_PARSER_H

#include <string>
#include <vector>
#include <map>

// Representing a single attribute in the dataset schema
struct Attribute {
    std::string name;  // Name of the attribute (e.g., Outlook, Temperature, in1, out1)
    bool isNumeric;  // True if attribute values are numeric (continuous)

    // List of possible categorical values
    // Example: {"Sunny", "Overcast", "Rain"}
    std::vector<std::string> categories;
};

// Class responsible for parsing the attribute schema file
class AttrParser {
public:
    // Constructor that stores the path to the attribute file
    AttrParser(const std::string& filename);

    // Reads and parses the attribute file to extract schema information
    void parse();  

    // Returns the list of parsed attributes
    const std::vector<Attribute>& getAttributes() const;

    // Returns the index of the class attribute
    // For classification datasets this is usually the last attribute
    int getClassIndex() const;

private:
    std::string filename;  // Path to the attribute schema file
    std::vector<Attribute> attributes;  // Stores the parsed attribute definitions
};

#endif