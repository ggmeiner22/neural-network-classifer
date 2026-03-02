#ifndef MLP_H
#define MLP_H

#include <vector>

class MLP {
public:
    enum OutputMode {
        CLASSIFICATION_SOFTMAX,  // tennis/iris
        MULTI_SIGMOID            // identity
    };

    MLP(int inputSize, int hiddenSize, int outputSize, OutputMode mode);

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
    int inSize, hidSize, outSize;
    OutputMode mode;

    std::vector<std::vector<double>> W1; // hidSize x inSize
    std::vector<std::vector<double>> W2; // outSize x hidSize

    static double sigmoid(double z);
    static std::vector<double> softmax(const std::vector<double>& v);

    // compute hidden activations
    std::vector<double> hiddenActivations(const std::vector<double>& x) const;
};

#endif