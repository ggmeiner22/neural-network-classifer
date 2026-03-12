#include "Util.h"
#include "MLP.h"

#include <algorithm>
#include <cstdlib>

namespace Util {

    // Returns the index of the largest value in vector v
    int argmax(const std::vector<double>& v) {
        // std::max_element returns an iterator to the largest element
        // std::distance converts the iterator position to an index
        return (int)std::distance(v.begin(), std::max_element(v.begin(), v.end()));
    }
    
    // Computes classification accuracy for a dataset
    double accuracyClass(const std::vector<std::vector<double>>& X,
                         const std::vector<std::vector<double>>& Y,
                         const MLP& model)
    {
        // If dataset is empty, accuracy is 0
        if (X.empty()) {
            return 0.0;
        }
        int correct = 0;  // Count correct predictions

        // Loop through each example
        for (size_t i = 0; i < X.size(); i++) {
            // Predict class using the neural network
            int pred = model.predictClass(X[i]);
            // Get the true class index from one-hot encoded label
            int truth = argmax(Y[i]);
            // If prediction matches the true label, count as correct
            if (pred == truth) {
                correct++;
            }
        }
        // Return accuracy as fraction of correct predictions
        return (double)correct / (double)X.size();
    }
    
    // Computes exact-match accuracy for identity mapping tasks
    double accuracyIdentityExact(const std::vector<std::vector<double>>& X,
                                 const std::vector<std::vector<double>>& Y,
                                 const MLP& model,
                                 double threshold)
    {
        // If dataset is empty, return 0
        if (X.empty()) {
            return 0.0;
        }
        int correctPatterns = 0;  // Count patterns predicted perfectly
    
        // Loop through each input pattern
        for (size_t i = 0; i < X.size(); i++) {

            // Get model output for the input
            std::vector<double> out = model.forward(X[i]);
    
            bool ok = true;  // Assume prediction is correct

            // Compare each output bit with target bit
            for (size_t k = 0; k < out.size(); k++) {

                // Convert output probability to binary using threshold
                int predBit = (out[k] >= threshold) ? 1 : 0;

                // Convert true output value to binary
                int trueBit = (Y[i][k] >= 0.5) ? 1 : 0;

                // If any bit differs, the pattern is incorrect
                if (predBit != trueBit) { 
                    ok = false; 
                    break; 
                }
            }
            // Count this pattern if all bits match
            if (ok) {
                correctPatterns++;
            }
        }
        // Return fraction of patterns that were perfectly predicted
        return (double)correctPatterns / (double)X.size();
    }
    
    // Splits dataset into training and validation subsets
    void splitTrainVal(const std::vector<std::vector<double>>& X,
                       const std::vector<std::vector<double>>& Y,
                       double valFrac,
                       std::vector<std::vector<double>>& Xtr,
                       std::vector<std::vector<double>>& Ytr,
                       std::vector<std::vector<double>>& Xval,
                       std::vector<std::vector<double>>& Yval)
    {
        // Clear any existing data
        Xtr.clear(); Ytr.clear(); Xval.clear(); Yval.clear();

        // If dataset is empty, nothing to split
        if (X.empty()) {
            return;
        }
    
        // Compute number of validation examples
        size_t nVal = (size_t)(valFrac * X.size());

        // Determine split index (last nVal samples go to validation set)
        size_t cut = (X.size() > nVal) ? (X.size() - nVal) : 0;
    
        // Copy examples into training or validation sets
        for (size_t i = 0; i < X.size(); i++) {
            if (i < cut) { 
                Xtr.push_back(X[i]); 
                Ytr.push_back(Y[i]); 
            } else { 
                Xval.push_back(X[i]); 
                Yval.push_back(Y[i]); 
            }
        }
    }
    
    // Corrupts a percentage of labels in a one-hot encoded dataset
    std::vector<std::vector<double>> corruptOneHotLabels(const std::vector<std::vector<double>>& Y,
                                                         double noisePercent,
                                                         unsigned seed)
    {
        // Make a copy of the original labels
        std::vector<std::vector<double>> Yn = Y;

        // Ensure not empty
        if (Yn.empty()) {
            return Yn;
        }
    
        // Number of classes
        int K = (int)Yn[0].size();

        // No corruption possible if only one class
        if (K <= 1) {
            return Yn;
        }
    
        // Compute how many labels to corrupt
        size_t nCorrupt = (size_t)((noisePercent / 100.0) * (double)Yn.size() + 0.5);
    
        std::srand(seed);  // Seed the random number generator
        
        // Track which samples have already been corrupted
        std::vector<int> used(Yn.size(), 0);
        size_t done = 0;
    
        // Continue until required number of labels are corrupted
        while (done < nCorrupt) {

            // Choose random example index
            size_t idx = (size_t)(std::rand() % (int)Yn.size());

            // Skip if already modified
            if (used[idx]) {
                continue;
            }
            used[idx] = 1;
    
            // Get original class index
            int oldC = argmax(Yn[idx]);
            int newC = oldC;

            // Choose a different class randomly
            while (newC == oldC) {
                newC = std::rand() % K;
            }
    
            // Reset the one-hot vector
            for (int j = 0; j < K; j++) {
                Yn[idx][j] = 0.0;
            }
            // Set the new corrupted class
            Yn[idx][newC] = 1.0;
    
            done++;
        }    
        // Return dataset with corrupted labels
        return Yn;
    }
} 