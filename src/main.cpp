#include <iostream>
#include <iomanip>
#include <string>
#include <cstdlib>
#include <sstream>

#include "Dataset.h"
#include "MLP.h"
#include "Util.h"

struct Config {
    std::string mode;       // identity | tennis | iris | iris_noisy
    std::string attrPath;
    std::string trainPath;
    std::string testPath;   // required for tennis/iris/iris_noisy
    int hidden = 16;
    double lr = 0.01;
    double momentum = 0.0;
    int epochs = 1000;
    double valFrac = 0.2;   // for iris_noisy "use validation"
    unsigned seed = 1;      // for corruption reproducibility
    double weightDecay = 0.0;
};

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

static bool isFlag(const std::string& s) {
    return s.size() >= 2 && s[0] == '-' && s[1] == '-';
}

static bool parseArgs(int argc, char* argv[], Config& c) {
    if (argc < 2) return false;

    for (int i = 1; i < argc; i++) {
        std::string key = argv[i];

        auto need = [&](const char* k)->std::string{
            if (i + 1 >= argc) { std::cerr << "Missing value after " << k << "\n"; return ""; }
            std::string v = argv[++i];
            if (isFlag(v)) { std::cerr << "Missing value after " << k << "\n"; return ""; }
            return v;
        };

        if (key == "--mode") c.mode = need("--mode");
        else if (key == "--attr") c.attrPath = need("--attr");
        else if (key == "--train") c.trainPath = need("--train");
        else if (key == "--test") c.testPath = need("--test");
        else if (key == "--hidden") c.hidden = std::atoi(need("--hidden").c_str());
        else if (key == "--lr") c.lr = std::atof(need("--lr").c_str());
        else if (key == "--momentum") c.momentum = std::atof(need("--momentum").c_str());
        else if (key == "--epochs") c.epochs = std::atoi(need("--epochs").c_str());
        else if (key == "--valfrac") c.valFrac = std::atof(need("--valfrac").c_str());
        else if (key == "--seed") c.seed = (unsigned)std::atoi(need("--seed").c_str());
        else if (key == "--weight_decay") c.weightDecay = std::atof(need("--weight_decay").c_str());
        else if (key == "--help" || key == "-h") return false;
        else { std::cerr << "Unknown flag: " << key << "\n"; return false; }
    }

    if (c.mode.empty() || c.attrPath.empty() || c.trainPath.empty()) return false;
    if (c.hidden <= 0 || c.epochs <= 0 || c.lr <= 0.0) return false;
    if (c.momentum < 0.0 || c.momentum >= 1.0) return false;

    // test required for non-identity modes (except if you decide otherwise)
    if ((c.mode == "tennis" || c.mode == "iris" || c.mode == "iris_noisy") && c.testPath.empty())
        return false;

    return true;
}

static std::string bitsStringFromVector(const std::vector<double>& v, double threshold = 0.5) {
    std::string s;
    s.reserve(v.size());
    for (size_t i = 0; i < v.size(); i++) {
        s.push_back((v[i] >= threshold) ? '1' : '0');
    }
    return s;
}

static void runIdentity(const Config& cfg, int hiddenUnits) {
    Dataset train(cfg.attrPath, cfg.trainPath);
    train.load();

    MLP model(train.inputSize(), hiddenUnits, train.outputSize(), MLP::MULTI_SIGMOID);
    model.train(train.X(), train.Y(), cfg.epochs, cfg.lr, cfg.momentum, cfg.weightDecay);

    double acc = Util::accuracyIdentityExact(train.X(), train.Y(), model, 0.5);

    std::cout << "=== testIdentity (hidden=" << hiddenUnits << ") ===\n";
    std::cout << "Training exact-match accuracy (threshold 0.5): " << std::fixed << std::setprecision(4) << acc << "\n\n";

    // Per input: hidden values + binary, and output values
    for (size_t i = 0; i < train.X().size(); i++) {
    std::vector<double> h, o;
    model.forwardWithHidden(train.X()[i], h, o);

    std::string inBits = bitsStringFromVector(train.X()[i], 0.5);
    std::string outBits = bitsStringFromVector(o, 0.5);

    std::cout << std::left << std::setw(10) << inBits << "-> ";

    // ---- Hidden values (2 decimals, aligned) ----
    std::cout << std::fixed << std::setprecision(2);
    for (size_t k = 0; k < h.size(); k++) {
        std::ostringstream tmp;
        tmp << std::fixed << std::setprecision(2) << h[k];
        std::string s = tmp.str();
        std::cout << std::setw(5) << s;
    }

    // ---- Hidden binary ----
    std::cout << " (";
    for (size_t k = 0; k < h.size(); k++) {
        std::cout << (h[k] >= 0.5 ? 1 : 0);
        if (k + 1 < h.size()) std::cout << " ";
    }
    std::cout << ") -> ";

    // ---- Output values (1 decimal, aligned) ----
    std::cout << std::fixed << std::setprecision(1);
    for (size_t k = 0; k < o.size(); k++) {
        std::cout << std::setw(5) << o[k];
    }

    std::cout << "  (" << outBits << ")";
    std::cout << "\n";
}
    std::cout << "\n";
}

static void runClassification(const Config& cfg, const std::string& title) {
    Dataset train(cfg.attrPath, cfg.trainPath);
    Dataset test(cfg.attrPath, cfg.testPath);
    train.load();
    test.load();

    MLP model(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);
    model.train(train.X(), train.Y(), cfg.epochs, cfg.lr, cfg.momentum, cfg.weightDecay);

    double accTr = Util::accuracyClass(train.X(), train.Y(), model);
    double accTe = Util::accuracyClass(test.X(), test.Y(), model);

    std::cout << "=== " << title << " ===\n";
    std::cout << "Train accuracy: " << std::fixed << std::setprecision(4) << accTr << "\n";
    std::cout << "Test  accuracy: " << std::fixed << std::setprecision(4) << accTe << "\n\n";
}

static void runIrisNoisy(const Config& cfg) {
    Dataset train(cfg.attrPath, cfg.trainPath);
    Dataset test(cfg.attrPath, cfg.testPath);
    train.load();
    test.load();

    std::cout << "=== testIrisNoisy ===\n";
    std::cout << "Noise%  TestAcc(no-val)  TestAcc(with-val)\n";

    for (int noise = 0; noise <= 20; noise += 2) {
        // corrupt training labels (one-hot)
        std::vector<std::vector<double>> Ynoisy = Util::corruptOneHotLabels(train.Y(), (double)noise, cfg.seed);

        // ---- no validation: train full epochs ----
        {
            MLP m(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);
            m.train(train.X(), Ynoisy, cfg.epochs, cfg.lr, cfg.momentum, cfg.weightDecay);
            double accTe = Util::accuracyClass(test.X(), test.Y(), m);

            std::cout << std::setw(5) << noise << "   "
                      << std::fixed << std::setprecision(4) << std::setw(13) << accTe;
        }

        // ---- with validation: simple early stopping on val accuracy ----
        {
            std::vector<std::vector<double>> Xtr, Ytr, Xval, Yval;
            Util::splitTrainVal(train.X(), Ynoisy, cfg.valFrac, Xtr, Ytr, Xval, Yval);

            MLP best(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);
            double bestVal = -1.0;

            // train in chunks so we can early-stop-ish
            int chunk = 50;
            int maxEpochs = cfg.epochs;
            int patience = 10; // chunks
            int bad = 0;

            MLP current(train.inputSize(), cfg.hidden, train.outputSize(), MLP::CLASSIFICATION_SOFTMAX);

            for (int e = 0; e < maxEpochs; e += chunk) {
                int steps = (e + chunk <= maxEpochs) ? chunk : (maxEpochs - e);
                current.train(Xtr, Ytr, steps, cfg.lr, cfg.momentum, cfg.weightDecay);

                double valAcc = Util::accuracyClass(Xval, Yval, current);
                if (valAcc > bestVal + 1e-12) {
                    bestVal = valAcc;
                    best = current; // copy weights
                    bad = 0;
                } else {
                    bad++;
                    if (bad >= patience) break;
                }
            }

            double accTe = Util::accuracyClass(test.X(), test.Y(), best);
            std::cout << "   " << std::fixed << std::setprecision(4) << std::setw(15) << accTe << "\n";
        }
    }

    std::cout << "\n";
}

int main(int argc, char* argv[]) {
    Config cfg;
    if (!parseArgs(argc, argv, cfg)) {
        usage(argv[0]);
        return 1;
    }

    // Use the provided seed for all randomness (weights, shuffling, etc.).
    // Previously MLP constructor re-seeded with time(0), making runs
    // nondeterministic; now we rely on this call so the same seed yields the
    // same results every invocation.
    std::srand(cfg.seed);

    if (cfg.mode == "identity") {
        // if the user specified a hidden size, only run that one; otherwise
        // default behaviour used to run both 3 and 4 for convenience.  Grid
        // search depends on honouring the --hidden flag so that each call
        // corresponds to a single architecture.
        if (cfg.hidden > 0) {
            runIdentity(cfg, cfg.hidden);
        } else {
            runIdentity(cfg, 3);
            runIdentity(cfg, 4);
        }
        return 0;
    }

    if (cfg.mode == "tennis") {
        runClassification(cfg, "testTennis");
        return 0;
    }

    if (cfg.mode == "iris") {
        runClassification(cfg, "testIris");
        return 0;
    }

    if (cfg.mode == "iris_noisy") {
        runIrisNoisy(cfg);
        return 0;
    }

    std::cerr << "Unknown mode: " << cfg.mode << "\n";
    usage(argv[0]);
    return 1;
}