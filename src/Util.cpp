#include "Util.h"
#include "MLP.h"

#include <algorithm>
#include <cstdlib>

namespace Util {

int argmax(const std::vector<double>& v) {
    return (int)std::distance(v.begin(), std::max_element(v.begin(), v.end()));
}

double accuracyClass(const std::vector<std::vector<double>>& X,
                     const std::vector<std::vector<double>>& Y,
                     const MLP& model)
{
    if (X.empty()) return 0.0;
    int correct = 0;
    for (size_t i = 0; i < X.size(); i++) {
        int pred = model.predictClass(X[i]);
        int truth = argmax(Y[i]);
        if (pred == truth) correct++;
    }
    return (double)correct / (double)X.size();
}

double accuracyIdentityExact(const std::vector<std::vector<double>>& X,
                             const std::vector<std::vector<double>>& Y,
                             const MLP& model,
                             double threshold)
{
    if (X.empty()) return 0.0;
    int correctPatterns = 0;

    for (size_t i = 0; i < X.size(); i++) {
        std::vector<double> out = model.forward(X[i]);

        bool ok = true;
        for (size_t k = 0; k < out.size(); k++) {
            int predBit = (out[k] >= threshold) ? 1 : 0;
            int trueBit = (Y[i][k] >= 0.5) ? 1 : 0;
            if (predBit != trueBit) { ok = false; break; }
        }
        if (ok) correctPatterns++;
    }

    return (double)correctPatterns / (double)X.size();
}

void splitTrainVal(const std::vector<std::vector<double>>& X,
                   const std::vector<std::vector<double>>& Y,
                   double valFrac,
                   std::vector<std::vector<double>>& Xtr,
                   std::vector<std::vector<double>>& Ytr,
                   std::vector<std::vector<double>>& Xval,
                   std::vector<std::vector<double>>& Yval)
{
    Xtr.clear(); Ytr.clear(); Xval.clear(); Yval.clear();
    if (X.empty()) return;

    size_t nVal = (size_t)(valFrac * X.size());
    // deterministic: last nVal to validation
    size_t cut = (X.size() > nVal) ? (X.size() - nVal) : 0;

    for (size_t i = 0; i < X.size(); i++) {
        if (i < cut) { Xtr.push_back(X[i]); Ytr.push_back(Y[i]); }
        else         { Xval.push_back(X[i]); Yval.push_back(Y[i]); }
    }
}

std::vector<std::vector<double>> corruptOneHotLabels(const std::vector<std::vector<double>>& Y,
                                                     double noisePercent,
                                                     unsigned seed)
{
    std::vector<std::vector<double>> Yn = Y;
    if (Yn.empty()) return Yn;

    int K = (int)Yn[0].size();
    if (K <= 1) return Yn;

    // number to corrupt
    size_t nCorrupt = (size_t)((noisePercent / 100.0) * (double)Yn.size() + 0.5);

    std::srand(seed);
    // simple: corrupt first nCorrupt random indices (with replacement avoidance)
    std::vector<int> used(Yn.size(), 0);
    size_t done = 0;

    while (done < nCorrupt) {
        size_t idx = (size_t)(std::rand() % (int)Yn.size());
        if (used[idx]) continue;
        used[idx] = 1;

        int oldC = argmax(Yn[idx]);
        int newC = oldC;
        while (newC == oldC) newC = std::rand() % K;

        for (int j = 0; j < K; j++) Yn[idx][j] = 0.0;
        Yn[idx][newC] = 1.0;

        done++;
    }

    return Yn;
}

} 