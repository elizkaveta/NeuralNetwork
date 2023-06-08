#include "src/Model.h"
#include <vector>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;

int main() {
    std::vector<std::unique_ptr<ActivationFunction> > activation_functions;
    activation_functions.push_back(std::make_unique<ReLu>());
    activation_functions.push_back(std::make_unique<Softmax>());
    Model model({{784, 128, 10}, std::move(activation_functions)},
                std::make_unique<MSE>(), 0.05);
    for (int i = 0; i < 10; ++i) {
        model.Train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 1, 1);
        auto a = model.Predict("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 30);
        std::cout << a.first * 1.0 / a.second << std::endl;
    }
    std::ofstream file("out.txt");
    file << model;
    return 0;
}
