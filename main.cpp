//
// Created by Elizaveta Iashchinskaia on 11.05.2023.
//
#include "src/Model.h"
#include <vector>
#include <memory>


using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;
int main() {
    LinearLayer b(1, 1);
    std::vector<LinearLayer> c(1, b);
    Model model(Sequential({{1,2}}), Sigmoid(), MSE(), 0.1);
    model.Train();
}