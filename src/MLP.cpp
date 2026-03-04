#include "MLP.h"
#include <cstdlib>
#include <iostream>
#include <ctime>
#include <cmath>
#include <algorithm>

static double randWeight() {
    return (double)std::rand() / (double)RAND_MAX - 0.5;
}

MLP::MLP(int inputSize, int hiddenSize, int outputSize, OutputMode m)
    : inSize(inputSize), hidSize(hiddenSize), outSize(outputSize), mode(m)
{
    // weight initialization relies on global RNG state; seed should be set
    // once by the caller (main) using --seed.  Do not reseed here with time,
    // otherwise repeated runs produce different results even with the same
    // command-line arguments.

    W1.assign(hidSize, std::vector<double>(inSize, 0.0));
    W2.assign(outSize, std::vector<double>(hidSize, 0.0));

    for (int i = 0; i < hidSize; i++)
        for (int j = 0; j < inSize; j++)
            W1[i][j] = randWeight();

    for (int i = 0; i < outSize; i++)
        for (int j = 0; j < hidSize; j++)
            W2[i][j] = randWeight();
}

double MLP::sigmoid(double z) {
    return 1.0 / (1.0 + std::exp(-z));
}

std::vector<double> MLP::softmax(const std::vector<double>& v) {
    std::vector<double> out(v.size(), 0.0);
    double maxVal = *std::max_element(v.begin(), v.end());
    double sum = 0.0;
    for (size_t i = 0; i < v.size(); i++) {
        out[i] = std::exp(v[i] - maxVal);
        sum += out[i];
    }
    for (size_t i = 0; i < v.size(); i++) out[i] /= sum;
    return out;
}

std::vector<double> MLP::hiddenActivations(const std::vector<double>& x) const {
    std::vector<double> h(hidSize, 0.0);
    for (int i = 0; i < hidSize; i++) {
        double sum = 0.0;
        for (int j = 0; j < inSize; j++)
            sum += W1[i][j] * x[j];
        h[i] = sigmoid(sum);
    }
    return h;
}

void MLP::forwardWithHidden(const std::vector<double>& x,
                            std::vector<double>& hiddenOut,
                            std::vector<double>& outputOut) const
{
    hiddenOut = hiddenActivations(x);

    std::vector<double> logits(outSize, 0.0);
    for (int i = 0; i < outSize; i++) {
        double sum = 0.0;
        for (int j = 0; j < hidSize; j++)
            sum += W2[i][j] * hiddenOut[j];
        logits[i] = sum;
    }

    if (mode == CLASSIFICATION_SOFTMAX) {
        outputOut = softmax(logits);
    } else {
        // identity: per-output sigmoid
        outputOut.assign(outSize, 0.0);
        for (int i = 0; i < outSize; i++)
            outputOut[i] = sigmoid(logits[i]);
    }
}

std::vector<double> MLP::forward(const std::vector<double>& x) const {
    std::vector<double> h, out;
    forwardWithHidden(x, h, out);
    return out;
}

int MLP::predictClass(const std::vector<double>& x) const {
    std::vector<double> out = forward(x);
    int best = 0;
    for (int i = 1; i < (int)out.size(); i++)
        if (out[i] > out[best]) best = i;
    return best;
}


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

    for (int ep = 0; ep < epochs; ep++) {
        double totalLoss = 0.0;

        for (size_t n = 0; n < X.size(); n++) {
            const std::vector<double>& x = X[n];
            const std::vector<double>& y = Y[n];

            // ---- forward ----
            std::vector<double> h = hiddenActivations(x);

            std::vector<double> logits(outSize, 0.0);
            for (int i = 0; i < outSize; i++) {
                double sum = 0.0;
                for (int j = 0; j < hidSize; j++)
                    sum += W2[i][j] * h[j];
                logits[i] = sum;
            }

            std::vector<double> out(outSize, 0.0);
            if (mode == CLASSIFICATION_SOFTMAX) out = softmax(logits);
            else {
                for (int i = 0; i < outSize; i++) out[i] = sigmoid(logits[i]);
            }

            // ---- ADD: accumulate loss for convergence ----
            if (mode == CLASSIFICATION_SOFTMAX) {
                // Cross-entropy: -sum y_k log(out_k). For one-hot y, this is just -log(p_true)
                double sampleLoss = 0.0;
                for (int k = 0; k < outSize; k++) {
                    if (y[k] > 0.0) {
                        double p = out[k];
                        if (p < 1e-12) p = 1e-12; // avoid log(0)
                        sampleLoss += -std::log(p);
                    }
                }
                totalLoss += sampleLoss;
            } else {
                // Identity: MSE = 0.5 * sum (out_k - y_k)^2
                double sampleLoss = 0.0;
                for (int k = 0; k < outSize; k++) {
                    double diff = out[k] - y[k];
                    sampleLoss += 0.5 * diff * diff;
                }
                totalLoss += sampleLoss;
            }

            // ---- output delta ----
            std::vector<double> deltaOut(outSize, 0.0);

            if (mode == CLASSIFICATION_SOFTMAX) {
                // softmax + cross entropy: out - y
                for (int i = 0; i < outSize; i++)
                    deltaOut[i] = out[i] - y[i];
            } else {
                // identity: squared error w/ sigmoid derivative
                for (int i = 0; i < outSize; i++) {
                    double s = out[i];
                    deltaOut[i] = (s - y[i]) * (s * (1.0 - s));
                }
            }

            // ---- hidden delta ----
            std::vector<double> deltaH(hidSize, 0.0);
            for (int j = 0; j < hidSize; j++) {
                double sum = 0.0;
                for (int i = 0; i < outSize; i++)
                    sum += W2[i][j] * deltaOut[i];
                deltaH[j] = sum * (h[j] * (1.0 - h[j]));
            }

            // ---- update W2 with momentum + optional weight decay ----
            for (int i = 0; i < outSize; i++) {
                for (int j = 0; j < hidSize; j++) {
                    double grad = deltaOut[i] * h[j];
                    if (weightDecay > 0.0) grad += weightDecay * W2[i][j];

                    V2[i][j] = momentum * V2[i][j] + lr * grad;
                    W2[i][j] -= V2[i][j];
                }
            }

            // ---- update W1 with momentum + optional weight decay ----
            for (int j = 0; j < hidSize; j++) {
                for (int k = 0; k < inSize; k++) {
                    double grad = deltaH[j] * x[k];
                    if (weightDecay > 0.0) grad += weightDecay * W1[j][k];

                    V1[j][k] = momentum * V1[j][k] + lr * grad;
                    W1[j][k] -= V1[j][k];
                }
            }
        }
//
        /* ---- ADD: convergence check at end of epoch ----
        double avgLoss = totalLoss / (double)X.size();

        if (bestLoss - avgLoss > minDelta) {
            bestLoss = avgLoss;
            stall = 0;
        } else {
            stall++;
            if (stall >= patience) {
                std::cout << "Converged at epoch " << ep
                          << " avgLoss=" << avgLoss << "\n";
                break; // converged
            }
        }*/
    }
}