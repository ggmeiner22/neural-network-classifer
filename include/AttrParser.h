#ifndef ATTR_PARSER_H
#define ATTR_PARSER_H

#include <string>
#include <vector>
#include <map>

struct Attribute {
    std::string name;
    bool isNumeric;
    std::vector<std::string> categories;
};

class AttrParser {
public:
    AttrParser(const std::string& filename);
    void parse();

    const std::vector<Attribute>& getAttributes() const;
    int getClassIndex() const;

private:
    std::string filename;
    std::vector<Attribute> attributes;
};

#endif