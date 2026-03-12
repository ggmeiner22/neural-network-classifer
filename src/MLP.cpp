#include "MLP.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>

// Generates a random weight in the range [-0.5, 0.5]
static double randWeight() {
    return (double)std::rand() / (double)RAND_MAX - 0.5;
}

// Constructor for the Multi-Layer Perceptron
MLP::MLP(int inputSize, int hiddenSize, int outputSize, OutputMode m)
    : inSize(inputSize), hidSize(hiddenSize), outSize(outputSize), mode(m)
{
    
    // Weight initialization relies on the global random number generator.
    // The random seed should be set once in main() using std::srand(seed).

    // Initialize input -> hidden weight matrix
    W1.assign(hidSize, std::vector<double>(inSize, 0.0));

    // Initialize hidden -> output weight matrix
    W2.assign(outSize, std::vector<double>(hidSize, 0.0));

    // Fill W1 with random weights
    for (int i = 0; i < hidSize; i++) {
        for (int j = 0; j < inSize; j++) {
            W1[i][j] = randWeight();
        }
    }

    // Fill W2 with random weights
    for (int i = 0; i < outSize; i++) {
        for (int j = 0; j < hidSize; j++) {
            W2[i][j] = randWeight();
        }
    }
}

// Sigmoid activation function
double MLP::sigmoid(double z) {
    // σ(z) = 1 / (1 + e^-z)
    return 1.0 / (1.0 + std::exp(-z));
}

// Softmax activation function for multi-class classification
std::vector<double> MLP::softmax(const std::vector<double>& v) {
    std::vector<double> out(v.size(), 0.0);

    // Find the maximum value
    double maxVal = *std::max_element(v.begin(), v.end());
    double sum = 0.0;

    // Compute exponentials
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = std::exp(v[i] - maxVal);
        sum += out[i];
    }
    // Normalize so probabilities sum to 1
    for (size_t i = 0; i < v.size(); i++) {
        out[i] /= sum;
    }
    return out;
}

// Computes hidden layer activations for a given input vector
std::vector<double> MLP::hiddenActivations(const std::vector<double>& x) const {
    std::vector<double> h(hidSize, 0.0);

    // For each hidden neuron
    for (int i = 0; i < hidSize; i++) {
        double sum = 0.0;

        // Weighted sum of inputs
        for (int j = 0; j < inSize; j++) {
            sum += W1[i][j] * x[j];
        }
        // Apply sigmoid activation
        h[i] = sigmoid(sum);
    }
    return h;
}

// Performs forward propagation while also returning hidden activations
void MLP::forwardWithHidden(const std::vector<double>& x,
                            std::vector<double>& hiddenOut,
                            std::vector<double>& outputOut) const
{
    // Compute hidden layer
    hiddenOut = hiddenActivations(x);

    // Compute raw output layer values (logits)
    std::vector<double> logits(outSize, 0.0);
    // For each hidden neuron
    for (int i = 0; i < outSize; i++) {
        double sum = 0.0;
        // Weighted sum of inputs
        for (int j = 0; j < hidSize; j++) {
            sum += W2[i][j] * hiddenOut[j];
        }
        logits[i] = sum;
    }

    // Apply appropriate output activation
    if (mode == CLASSIFICATION_SOFTMAX) {
        // For classification tasks
        outputOut = softmax(logits);
    } else {
        // Identity mode uses independent sigmoid outputs
        outputOut.assign(outSize, 0.0);
        for (int i = 0; i < outSize; i++) {
            outputOut[i] = sigmoid(logits[i]);
        }
    }
}

// Standard forward pass (returns only output layer)
std::vector<double> MLP::forward(const std::vector<double>& x) const {
    std::vector<double> h, out;
    forwardWithHidden(x, h, out);
    return out;
}

// Predicts the class index with the highest output probability
int MLP::predictClass(const std::vector<double>& x) const {
    std::vector<double> out = forward(x);
    int best = 0;
    // Find index of largest output probability
    for (int i = 1; i < (int)out.size(); i++) {
        if (out[i] > out[best]) best = i;
    }
    return best;
}

// Train the neural network using stochastic gradient descent + momentum
void MLP::train(const std::vector<std::vector<double>>& X,
                const std::vector<std::vector<double>>& Y,
                int epochs,
                double lr,
                double momentum,
                double weightDecay)
{
    // Momentum velocities
    std::vector<std::vector<double>> V1(hidSize, std::vector<double>(inSize, 0.0));
    std::vector<std::vector<double>> V2(outSize, std::vector<double>(hidSize, 0.0));

    // Train the network until stopping criteron: Epochs
    for (int ep = 0; ep < epochs; ep++) {
        double totalLoss = 0.0;  // Accumulate total loss over this epoch

        // For each training example
        for (size_t n = 0; n < X.size(); n++) {
            // Get the current input vector and target output vector
            const std::vector<double>& x = X[n];
            const std::vector<double>& y = Y[n];

            // ---- forward pass ----

            // Compute activations of the hidden layer
            std::vector<double> h = hiddenActivations(x);

            // Store the raw output-layer values before activation
            std::vector<double> logits(outSize, 0.0);

            // Compute each output neuron's weighted sum from the hidden layer
            for (int i = 0; i < outSize; i++) {
                double sum = 0.0;
                for (int j = 0; j < hidSize; j++) {
                    sum += W2[i][j] * h[j];
                }
                logits[i] = sum;
            }

            // Store the final output activations
            std::vector<double> out(outSize, 0.0);

            // For classification mode, apply softmax to produce class probabilities
            if (mode == CLASSIFICATION_SOFTMAX) {
                out = softmax(logits);
            } else {
                // For identity mode, apply sigmoid independently to each output neuron
                for (int i = 0; i < outSize; i++) {
                    out[i] = sigmoid(logits[i]);
                }
            }

            // ---- compute loss ----

            // For classification mode, use cross-entropy loss
            if (mode == CLASSIFICATION_SOFTMAX) {
                double sampleLoss = 0.0;  // Loss for this one sample

                // Since y is one-hot encoded, only the true class contributes to loss
                for (int k = 0; k < outSize; k++) {
                    if (y[k] > 0.0) {
                        double p = out[k];
                        if (p < 1e-12) {
                            p = 1e-12; // avoid log(0)
                        }
                        sampleLoss += -std::log(p);
                    }
                }
                totalLoss += sampleLoss;  // Add this sample's loss to the epoch total
            } else {
                // For identity mode, use mean squared error loss
                double sampleLoss = 0.0;  // Loss for this one sample
                for (int k = 0; k < outSize; k++) {
                    double diff = out[k] - y[k];
                    sampleLoss += 0.5 * diff * diff;
                }
                // Add this sample's loss to the epoch total
                totalLoss += sampleLoss;
            }

            // ---- compute output layer delta ----

            // deltaOut stores the error signal for each output neuron
            std::vector<double> deltaOut(outSize, 0.0);

            if (mode == CLASSIFICATION_SOFTMAX) {
                // For softmax + cross-entropy, derivative simplifies to (out - y)
                for (int i = 0; i < outSize; i++) {
                    deltaOut[i] = out[i] - y[i];
                }
            } else {
                // For sigmoid + squared error, multiply by sigmoid derivative
                for (int i = 0; i < outSize; i++) {
                    double s = out[i];
                    deltaOut[i] = (s - y[i]) * (s * (1.0 - s));
                }
            }

            // ---- compute hidden layer delta ----

            // deltaH stores the error signal for each hidden neuron
            std::vector<double> deltaH(hidSize, 0.0);
            for (int j = 0; j < hidSize; j++) {
                double sum = 0.0;

                // Backpropagate output errors into hidden neuron j
                for (int i = 0; i < outSize; i++) {
                    sum += W2[i][j] * deltaOut[i];
                }
                // Multiply by derivative of sigmoid activation at hidden neuron j
                deltaH[j] = sum * (h[j] * (1.0 - h[j]));
            }

            // ---- update hidden-to-output weights W2 ----
            for (int i = 0; i < outSize; i++) {
                for (int j = 0; j < hidSize; j++) {
                    // Gradient of loss with respect to W2[i][j]
                    double grad = deltaOut[i] * h[j];

                    // Add L2 weight decay term if enabled
                    if (weightDecay > 0.0) {
                        grad += weightDecay * W2[i][j];
                    }

                    // Update velocity using momentum
                    V2[i][j] = momentum * V2[i][j] + lr * grad;
                    W2[i][j] -= V2[i][j];  // Gradient descent step
                }
            }

            // ---- update input-to-hidden weights W1 ----
            for (int j = 0; j < hidSize; j++) {
                for (int k = 0; k < inSize; k++) {
                    // Gradient of loss with respect to W1[j][k]
                    double grad = deltaH[j] * x[k];

                    // Add L2 weight decay term if enabled
                    if (weightDecay > 0.0) {
                        grad += weightDecay * W1[j][k];
                    }

                    // Update velocity using momentum
                    V1[j][k] = momentum * V1[j][k] + lr * grad;
                    W1[j][k] -= V1[j][k]; // Gradient descent step
                }
            }
        }
    }
}