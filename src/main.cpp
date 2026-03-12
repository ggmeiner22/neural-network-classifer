#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>

#include "Dataset.h"
#include "MLP.h"
#include "Util.h"

// Stores all runtime configuration parameters
struct Config {
    std::string mode;       // identity | tennis | iris | iris_noisy
    std::string attrPath;   // Path to attribute schema file
    std::string trainPath;  // Path to training dataset
    std::string testPath;   // required for tennis/iris/iris_noisy

    int hidden = 16;        // Number of hidden neurons in the neural network
    double lr = 0.01;       // Learning rate used in gradient descent
    double momentum = 0.0;  // Momentum coefficient
    int epochs = 1000;      // Number of training epochs
    double weightDecay = 0.0;  // Weight decay rate

    double valFrac = 0.2;   // Fraction of training data used as validation (for noisy experiments)
    unsigned seed = 1;      // Random seed for reproducibility
    
};

// Prints command-line usage instructions
static void usage(const char* prog) {
    std::cout
        << "Usage:\n"
        << "  " << prog << " --mode <identity|tennis|iris|iris_noisy>\n"
        << "        --attr <path> --train <path> [--test <path>]\n"
        << "        --hidden N --lr X --momentum M --epochs E\n"
        << "        [--valfrac F] [--seed S] [--weight_decay WD]\n\n"
        << "Example (identity):\n"
        << "  " << prog << " --mode identity --attr data/identity-attr.txt --train data/identity-train.txt \\\n"
        << "       --hidden 3 --lr 0.2 --momentum 0.9 --epochs 5000\n";
}

// Checks whether a command line argument is a flag EX: (--something)
static bool isFlag(const std::string& s) {
    return s.size() >= 2 && s[0] == '-' && s[1] == '-';
}

// Parses command-line arguments into the Config struct
static bool parseArgs(int argc, char* argv[], Config& c) {
    if (argc < 2) {  // Need at least one argument
        return false;
    }

    // Loop through command line arguments
    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];

        // Lambda function that retrieves the value following a flag
        auto need = [&](const char* k)->std::string{
            if (i + 1 >= argc) { 
                std::cerr << "Missing value after " << k << "\n"; 
                return ""; 
            }
            std::string v = argv[++i];
            if (isFlag(v)) { 
                std::cerr << "Missing value after " << k << "\n"; 
                return ""; 
            }
            return v;
        };

        // Match known command-line flags
        if (key == "--mode") {
            c.mode = need("--mode");
        } else if (key == "--attr") {
            c.attrPath = need("--attr");
        } else if (key == "--train") {
            c.trainPath = need("--train");
        } else if (key == "--test") {
            c.testPath = need("--test");
        } else if (key == "--hidden") {
            c.hidden = std::atoi(need("--hidden").c_str());
        } else if (key == "--lr") {
            c.lr = std::atof(need("--lr").c_str());
        } else if (key == "--momentum") {
            c.momentum = std::atof(need("--momentum").c_str());
        } else if (key == "--epochs") {
            c.epochs = std::atoi(need("--epochs").c_str());
        } else if (key == "--valfrac") {
            c.valFrac = std::atof(need("--valfrac").c_str());
        } else if (key == "--seed") {
            c.seed = (unsigned)std::atoi(need("--seed").c_str());
        } else if (key == "--weight_decay") {
            c.weightDecay = std::atof(need("--weight_decay").c_str());
        } else if (key == "--help" || key == "-h") {
            return false;
        } else { 
            std::cerr << "Unknown flag: " << key << "\n"; 
            return false; 
        }
    }

    // Ensure a mode, attribute file, and training file are present
    if (c.mode.empty() || c.attrPath.empty() || c.trainPath.empty()) {
        return false;
    }
    // Ensure hidden units, stopping criterion, and learning rate are valid numbers (> 0)
    if (c.hidden <= 0 || c.epochs <= 0 || c.lr <= 0.0) {
        return false;
    }
    // Ensure momentum is between 0 and 1
    if (c.momentum < 0.0 || c.momentum >= 1.0) {
        return false;
    }

    // test for non-identity modes
    if ((c.mode == "tennis" || c.mode == "iris" || c.mode == "iris_noisy") && c.testPath.empty())
        return false;

    return true;
}

// Converts a vector of probabilities into a binary string
static std::string bitsStringFromVector(const std::vector<double>& v, double threshold = 0.5) {
    std::string s;
    s.reserve(v.size());

    // Convert each value into 0 or 1
    for (size_t i = 0; i < v.size(); i++) {
        s.push_back((v[i] >= threshold) ? '1' : '0');
    }
    return s;
}

// Runs the identity mapping experiment
static void runIdentity(const Config& cfg, int hiddenUnits) {

    // Load training dataset using the provided attribute and training file paths
    Dataset train(cfg.attrPath, cfg.trainPath);
    train.load();  // Load training dataset

    // Create neural network
    MLP model(train.inputSize(), hiddenUnits, train.outputSize(), MLP::MULTI_SIGMOID);

    // Train network
    model.train(train.X(), train.Y(), cfg.epochs, cfg.lr, cfg.momentum, cfg.weightDecay);

    // Compute training accuracy
    double acc = Util::accuracyIdentityExact(train.X(), train.Y(), model, 0.5);

    std::cout << "=== testIdentity (hidden=" << hiddenUnits << ") ===\n";
    std::cout << "Training exact-match accuracy (threshold 0.5): " << std::fixed << std::setprecision(4) << acc << "\n\n";

    // Print predictions for each example
    for (size_t i = 0; i < train.X().size(); i++) {
        std::vector<double> h, o;

        // Forward pass retrieving hidden and output activations
        model.forwardWithHidden(train.X()[i], h, o);

        std::string inBits = bitsStringFromVector(train.X()[i], 0.5);
        std::string outBits = bitsStringFromVector(o, 0.5);

        std::cout << std::left << std::setw(10) << inBits << "-> ";

        // Print hidden activations
        std::cout << std::fixed << std::setprecision(2);
        for (size_t k = 0; k < h.size(); k++) {
            std::ostringstream tmp;
            tmp << std::fixed << std::setprecision(2) << h[k];
            std::string s = tmp.str();
            std::cout << std::setw(5) << s;
        }
    
        // Print hidden binary representation
        std::cout << " (";
        for (size_t k = 0; k < h.size(); k++) {
            std::cout << (h[k] >= 0.5 ? 1 : 0);
            if (k + 1 < h.size()) {
                std::cout << " ";
            }
        }
        std::cout << ") -> ";
    
        // Print output activations
        std::cout << std::fixed << std::setprecision(1);
        for (size_t k = 0; k < o.size(); k++) {
            std::cout << std::setw(5) << o[k];
        }
    
        std::cout << "  (" << outBits << ")";
        std::cout << "\n";
}
    std::cout << "\n";
}

// Runs a standard classification experiment -> used for tennis and iris
static void runClassification(const Config& cfg, const std::string& title) {

    // Load training dataset using the provided attribute and training file paths
    Dataset train(cfg.attrPath, cfg.trainPath);
    // Load test dataset using the provided attribute and test file paths
    Dataset test(cfg.attrPath, cfg.testPath);
    train.load();  // Parse and load training data into feature/label matrices
    test.load();  // Parse and load test data into feature/label matrices

    // Create an MLP for multi-class classification using softmax output
    MLP model(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);

    // Train the model on the training dataset
    model.train(train.X(), train.Y(), cfg.epochs, cfg.lr, cfg.momentum, cfg.weightDecay);

    // Compute classification accuracy on the training set
    double accTr = Util::accuracyClass(train.X(), train.Y(), model);

    // Compute classification accuracy on the test set
    double accTe = Util::accuracyClass(test.X(), test.Y(), model);

    // Print training and test accuracy
    std::cout << "=== " << title << " ===\n";
    std::cout << "Train accuracy: " << std::fixed << std::setprecision(4) << accTr << "\n";
    std::cout << "Test  accuracy: " << std::fixed << std::setprecision(4) << accTe << "\n\n";
}

// Runs the iris noisy-label experiment
static void runIrisNoisy(const Config& cfg) {

    // Load training dataset using the provided attribute and training file paths
    Dataset train(cfg.attrPath, cfg.trainPath);
    // Load test dataset using the provided attribute and test file paths
    Dataset test(cfg.attrPath, cfg.testPath);
    train.load();  // Parse and load training data into feature/label matrices
    test.load();  // Parse and load test data into feature/label matrices

    std::cout << "=== testIrisNoisy ===\n";
    std::cout << "Noise%  TestAcc(no-val)  TestAcc(with-val)\n";

    // Try label corruption levels from 0% to 20% in steps of 2%
    for (int noise = 0; noise <= 20; noise += 2) {
        // Create a noisy copy of the training labels by corrupting a percentage of one-hot labels
        std::vector<std::vector<double>> Ynoisy = Util::corruptOneHotLabels(train.Y(), (double)noise, cfg.seed);

        // no validation -> train full epochs
        {
            // Create a fresh classification MLP
            MLP m(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);
    
            // Train on the full noisy training set for all requested epochs
            m.train(train.X(), Ynoisy, cfg.epochs, cfg.lr, cfg.momentum, cfg.weightDecay);
            // Evaluate test accuracy after full training
            double accTe = Util::accuracyClass(test.X(), test.Y(), m);
    
            // Print noise percentage and the test accuracy without validation
            std::cout << std::setw(5) << noise << "   "
                      << std::fixed << std::setprecision(4) << std::setw(13) << accTe;
        }

        // with validation -> early stopping on val accuracy
        {
            // Split noisy training data into train and validation subsets
            std::vector<std::vector<double>> Xtr, Ytr, Xval, Yval;
            Util::splitTrainVal(train.X(), Ynoisy, cfg.valFrac, Xtr, Ytr, Xval, Yval);

            // This model will store the best-performing weights seen on validation data
            MLP best(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);

            // Track the best validation accuracy seen so far
            double bestVal = -1.0;

            // Train in chunks instead of all epochs at once to simulate early stopping
            int chunk = 50;   // Number of epochs per training chunk
            int maxEpochs = cfg.epochs;  // Maximum total number of training epochs
            int patience = 10;   // Stop after this many non-improving chunks
            int bad = 0;  // Counts consecutive chunks with no improvement

            // Current working model being trained incrementally
            MLP current(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);

            // Train in blocks of "chunk" epochs
            for (int e = 0; e < maxEpochs; e += chunk) {

                // Determine how many epochs to train in this block
                int steps = (e + chunk <= maxEpochs) ? chunk : (maxEpochs - e);
                // Train the current model for this chunk
                current.train(Xtr, Ytr, steps, cfg.lr, cfg.momentum, cfg.weightDecay);

                // Evaluate current model on validation data
                double valAcc = Util::accuracyClass(Xval, Yval, current);

                // If validation accuracy improved, save this model as the best so far
                if (valAcc > bestVal + 1e-12) {
                    bestVal = valAcc;
                    best = current;  // Copy current model weights into best model
                    bad = 0;  // Reset non-improvement counter
                } else {
                    bad++;  // Validation did not improve
                    if (bad >= patience) {  // Stop early if patience is exhausted
                        break;
                    }
                }
            }
            // Evaluate the best validation-selected model on the test set
            double accTe = Util::accuracyClass(test.X(), test.Y(), best);
            // Print the test accuracy for the validation set or early-stopping version
            std::cout << "   " << std::fixed << std::setprecision(4) << std::setw(15) << accTe << "\n";
        }
    }
    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    Config cfg;  // Stores all parsed runtime settings

    // Parse command-line arguments; if invalid, print usage and exit with error
    if (!parseArgs(argc, argv, cfg)) {
        usage(argv[0]);
        return 1;
    }

    // Seed the random number generator so runs are reproducible when the same seed is used
    // This affects random initialization, shuffling, and any other rand()-based behavior
    std::srand(cfg.seed);

    // Identity-mode experiment
    if (cfg.mode == "identity") {\
        // If a specific hidden size was provided, run only that configuration
        if (cfg.hidden > 0) {
            runIdentity(cfg, cfg.hidden);
        } else {
            // Otherwise run default convenience cases with 3 and 4 hidden units
            runIdentity(cfg, cfg.hidden);
            runIdentity(cfg, 4);
        }
        return 0;
    }

    // Tennis classification experiment
    if (cfg.mode == "tennis") {
        runClassification(cfg, "testTennis");
        return 0;
    }

    // Iris classification experiment
    if (cfg.mode == "iris") {
        runClassification(cfg, "testIris");
        return 0;
    }

    // Iris noisy-label experiment
    if (cfg.mode == "iris_noisy") {
        runIrisNoisy(cfg);
        return 0;
    }

    // If mode is not recognized, print an error and usage instructions
    std::cerr << "Unknown mode: " << cfg.mode << "\n";
    usage(argv[0]);
    return 1;
}