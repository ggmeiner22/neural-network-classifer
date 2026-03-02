CXX = g++
CXXFLAGS = -std=c++11 -Wall -O2
INCLUDE = -Iinclude

SRC = src/main.cpp \
      src/Dataset.cpp \
      src/AttrParser.cpp \
      src/MLP.cpp \
      src/Util.cpp

OBJ = $(SRC:.cpp=.o)

all: nn

nn: $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o nn

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDE) -c $< -o $@

clean:
	rm -f src/*.o nn