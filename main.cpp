#include "src/Model.h"
#include <vector>
#include <memory>

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;

int main() {
  srand(43);
  DataLoader data_loader_train("../train/train-images.idx3-ubyte", "../train/train-labels.idx1-ubyte", 1);
  DataLoader data_loader_test("../test/t10k-images.idx3-ubyte", "../test/t10k-labels.idx1-ubyte", 32);
  std::vector<std::unique_ptr<ActivationFunction> > activation_functions;

  activation_functions.push_back(std::make_unique<ReLu>());
  activation_functions.push_back(std::make_unique<Softmax>());
  for (int i = 0; i < 10; ++i)
    for (int j = 0; j < 10; ++j)
      a[i][j] = 0;

  Sequential sequential({784, 128, 10}, std::move(activation_functions));
  Model model(std::move(sequential), std::make_unique<MSE>(), 0.05);
  model.Train(data_loader_train, data_loader_test, 200);
  return 0;
}
