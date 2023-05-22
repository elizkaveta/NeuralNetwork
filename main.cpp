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
    activation_functions.push_back(std::make_unique<ReLu>());
    //activation_functions.push_back(std::make_unique<ReLu>());
    //activation_functions.push_back(std::make_unique<Sigmoid>());

    /*Sequential sequential({784, 128, 64, 32, 10}, std::move(activation_functions));
    Model model(std::move(sequential), std::make_unique<MSE>(), 0.00024);
    DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 500);
    model.Train(data_loader_train, 10);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
    int count_right = 0, size = 0;
    Batch m = data_loader_test.Next();
    std::vector<int> a(10);
    while(!m.empty()) {
        count_right += model.Predict(m);
        size += m.size();
        m = data_loader_test.Next();
    }
    std::cout << std::endl;
    std::cout << count_right << " " << size;
    return 0;*/
    Sequential sequential({784, 40, 10}, std::move(activation_functions));
    Model model(std::move(sequential), std::make_unique<MSE>(), 0.0000000000001);
    DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 16);
    model.Train(data_loader_train, 50);
    DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
    int count_right = 0, size = 0;
    Batch m = data_loader_test.Next();
    std::vector<int> a(10);
    while(!m.empty()) {
        count_right += model.Predict(m, a);
        size += m.size();
        m = data_loader_test.Next();
    }
    for (int i = 0; i < 10; ++i) {
        std::cout << a[i] << " ";
    }

    std::cout << std::endl;
    std::cout << count_right << " " << size;
    return 0;

}

// 0.0001 - 1003
// 0.000001 - 1082
// 0.00000001 - 1084
// 0.00000001 + 2)40 - 1283
// 0.000001 + 2)40 - 1282
// 0.0000000001 + 2)40 - 1283
// 0.000000000001 + 2)40 - 1283
// 0.00001 + 2)40 - 1100
