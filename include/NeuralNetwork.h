#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H

#include <vector>

class NeuralNetwork {
public:
    NeuralNetwork(int inputSize, int hiddenSize, int outputSize);

    void train(const std::vector<std::vector<double>>& X,
               const std::vector<std::vector<double>>& y,
               int epochs,
               double learningRate);

    std::vector<double> predict(const std::vector<double>& input);

private:
    int inputSize;
    int hiddenSize;
    int outputSize;

    std::vector<std::vector<double>> W1;
    std::vector<std::vector<double>> W2;

    std::vector<double> sigmoid(const std::vector<double>& v);
    double sigmoid(double x);
};

#endif