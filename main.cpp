#include "src/Model.h"
#include <vector>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;

int main() {
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
    activation_functions.push_back(std::make_unique<Sigmoid>());
    activation_functions.push_back(std::make_unique<ReLu>());
    activation_functions.push_back(std::make_unique<Softmax>());
    Sequential sequential({784, 60, 40, 10}, std::move(activation_functions));
    Model model(std::move(sequential), std::make_unique<MSE>(), 1);
    DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 1);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
    model.Train(data_loader_train, data_loader_test, 100);
    return 0;
}

// 784, 40, 25, 10, si, re, so, 0.1, 5 epoch -- 8655
// 784, 40, 40, 10, si, re, so, 0.05, 10 epoch -- 9200
// 784, 60, 40, 10, si, re, so, 0.05, 10 epoch -- 9386