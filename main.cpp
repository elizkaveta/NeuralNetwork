#include "src/Model.h"
#include <vector>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;

int main() {
    srand(45);
    DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 1);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
    std::vector<std::unique_ptr<ActivationFunction> > activation_functions;
    activation_functions.push_back(std::make_unique<Sigmoid>());
    activation_functions.push_back(std::make_unique<ReLu>());
    activation_functions.push_back(std::make_unique<Softmax>());
    Sequential sequential({784, 40, 40, 10}, std::move(activation_functions));
    Model model(std::move(sequential), std::make_unique<MSE>(), 0.05);
    model.Train(data_loader_train, data_loader_test, 100);
    return 0;
}
// 42 - 0.8237
// 45 -
// 784, 40, 25, 10, si, re, so, 0.1, 5 epoch -- 8655
// 784, 40, 40, 10, si, re, so, 0.05, 10 epoch -- 9200
// 784, 60, 40, 10, si, re, so, 0.05, 10 epoch -- 9386
/*
 *
 * epoch: 1 / 10
accuracy test: 0.847100

epoch: 2 / 10
accuracy test: 0.863600

epoch: 3 / 10
accuracy test: 0.867100

 */

// 60 60 60 S S R So 0.03
// 60 60 S Re So 0.03
// 60 40 S Re So 0.03
// 13 - 0.9448, 1 - 0.8967