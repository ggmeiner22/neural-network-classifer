#include "NeuralNetwork.h"
#include <cmath>
#include <cstdlib>
#include <ctime>

NeuralNetwork::NeuralNetwork(int inputSize, int hiddenSize, int outputSize)
    : inputSize(inputSize),
      hiddenSize(hiddenSize),
      outputSize(outputSize)
{
    std::srand(std::time(0));

    W1.resize(hiddenSize, std::vector<double>(inputSize));
    W2.resize(outputSize, std::vector<double>(hiddenSize));

    for (auto& row : W1)
        for (auto& w : row)
            w = ((double) rand() / RAND_MAX) - 0.5;

    for (auto& row : W2)
        for (auto& w : row)
            w = ((double) rand() / RAND_MAX) - 0.5;
}

double NeuralNetwork::sigmoid(double x) {
    return 1.0 / (1.0 + std::exp(-x));
}

std::vector<double> NeuralNetwork::sigmoid(const std::vector<double>& v) {
    std::vector<double> result(v.size());
    for (size_t i = 0; i < v.size(); i++)
        result[i] = sigmoid(v[i]);
    return result;
}

std::vector<double> NeuralNetwork::predict(const std::vector<double>& input) {
    std::vector<double> hidden(hiddenSize, 0.0);

    for (int i = 0; i < hiddenSize; i++)
        for (int j = 0; j < inputSize; j++)
            hidden[i] += W1[i][j] * input[j];

    hidden = sigmoid(hidden);

    std::vector<double> output(outputSize, 0.0);

    for (int i = 0; i < outputSize; i++)
        for (int j = 0; j < hiddenSize; j++)
            output[i] += W2[i][j] * hidden[j];

    return sigmoid(output);
}

void NeuralNetwork::train(
    const std::vector<std::vector<double>>& X,
    const std::vector<std::vector<double>>& y,
    int epochs,
    double learningRate)
{
    for (int epoch = 0; epoch < epochs; epoch++) {
        for (size_t n = 0; n < X.size(); n++) {

            auto hidden = sigmoid(predict(X[n]));

            // You can expand here to full backprop if required
        }
    }
}