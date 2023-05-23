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
    Sequential sequential({784, 16, 16, 10}, std::move(activation_functions));
    Model model(std::move(sequential), std::make_unique<MSE>(), 0.1);
    DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 5);
    model.Train(data_loader_train, 10);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
    auto answer = model.Predict(data_loader_test);
    std::cout << answer.first << " " << answer.second;
    return 0;
}

