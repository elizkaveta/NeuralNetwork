#include "src/Model.h"
#include <vector>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;

int main() {
    DataLoader data_loader_train("../train/train-images.idx3-ubyte",
                                 "../train/train-labels.idx1-ubyte", 1);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte",
                                "../test/t10k-labels.idx1-ubyte", 32);
    std::vector<std::unique_ptr<ActivationFunction> > activation_functions;
    activation_functions.push_back(std::make_unique<ReLu>());
    activation_functions.push_back(std::make_unique<Softmax>());
    Model model({{784, 128, 10}, std::move(activation_functions)},
                std::make_unique<MSE>(), 0.05);
    model.Train(std::move(data_loader_train), std::move(data_loader_test), 10);
    return 0;
}
