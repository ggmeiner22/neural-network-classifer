#ifndef UTIL_H
#define UTIL_H

#include <vector>
#include <string>

class MLP;

namespace Util {

// classification accuracy (iris/tennis)
double accuracyClass(const std::vector<std::vector<double>>& X,
                     const std::vector<std::vector<double>>& Y,
                     const MLP& model);

// identity “pattern accuracy”: all output bits correct using 0.5 threshold
double accuracyIdentityExact(const std::vector<std::vector<double>>& X,
                             const std::vector<std::vector<double>>& Y,
                             const MLP& model,
                             double threshold = 0.5);

// split indices into train/val (simple deterministic split)
void splitTrainVal(const std::vector<std::vector<double>>& X,
                   const std::vector<std::vector<double>>& Y,
                   double valFrac,
                   std::vector<std::vector<double>>& Xtr,
                   std::vector<std::vector<double>>& Ytr,
                   std::vector<std::vector<double>>& Xval,
                   std::vector<std::vector<double>>& Yval);

// corrupt a percentage of one-hot class labels (train set) by flipping to a different class
std::vector<std::vector<double>> corruptOneHotLabels(const std::vector<std::vector<double>>& Y,
                                                     double noisePercent,
                                                     unsigned seed);

// formatting helpers
int argmax(const std::vector<double>& v);

} // namespace Util

#endif