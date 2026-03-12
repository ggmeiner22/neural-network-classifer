#ifndef MLP_H
#define MLP_H

#include <vector>

// Class implementing a two-layer Multi-Layer Perceptron neural network
class MLP {
public:

    // Defines how the output layer should behave depending on the task
    enum OutputMode {
        CLASSIFICATION_SOFTMAX,  // Used for classification problems (tennis, iris)
                                 // Outputs probabilities using softmax activation

        MULTI_SIGMOID            // Used for identity mapping tasks
                                 // Each output neuron uses independent sigmoid activation
    };

    // Constructor: creates a neural network with specified layer sizes
    MLP(int inputSize, int hiddenSize, int outputSize, OutputMode mode);

    // Trains the network using backpropagation
    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& Y,
               int epochs,
               double learningRate,
               double momentum,
               double weightDecay = 0.0);

    // Forward that returns ONLY output probabilities
    std::vector<double> forward(const std::vector<double>& x) const;

    // Forward that ALSO returns hidden activations (for Figure 4.7-style printing)
    void forwardWithHidden(const std::vector<double>& x,
                           std::vector<double>& hiddenOut,
                           std::vector<double>& outputOut) const;

    // Classification helper (argmax of forward())
    int predictClass(const std::vector<double>& x) const;

private:
    // Number of neurons in each layer
    int inSize, hidSize, outSize;
    OutputMode mode;  // Determines whether the network uses softmax or sigmoid outputs

    // Weight matrix from input layer -> hidden layer
    // Dimensions: hiddenSize x inputSize
    std::vector<std::vector<double>> W1;

    // Weight matrix from hidden layer -> output layer
    // Dimensions: outputSize x hiddenSize
    std::vector<std::vector<double>> W2; // outSize x hidSize

    static double sigmoid(double z);  // Sigmoid activation function

    // Softmax activation function for classification outputs
    static std::vector<double> softmax(const std::vector<double>& v);

    // Computes hidden layer activations during forward propagation
    std::vector<double> hiddenActivations(const std::vector<double>& x) const;
};

#endif