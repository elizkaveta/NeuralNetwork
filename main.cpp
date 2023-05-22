#include "src/Model.h"
#include <vector>
#include <memory>
#include "src/SaveModel.h"
#include "src/DataLoader.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;

int main() {
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;

    activation_functions.push_back(std::make_unique<Sigmoid>());
    activation_functions.push_back(std::make_unique<Sigmoid>());

    Sequential sequential({784, 15, 10}, std::move(activation_functions));
    Model model(std::move(sequential), std::make_unique<MSE>(), 0.00001);
    DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 32);
    model.Train(data_loader_train, 20);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
    int count_right = 0, size = 0;
    Batch m = data_loader_test.Next();
    while(!m.empty()) {
        count_right += model.Predict(m);
        size += m.size();
        m = data_loader_test.Next();
    }
    std::cout << count_right << " " << size;
    return 0;

}