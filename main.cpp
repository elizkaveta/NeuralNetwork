//
// Created by Elizaveta Iashchinskaia on 11.05.2023.
//
#include "src/Model.h"
#include <vector>
#include <memory>
#include "src/SaveModel.h"

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using namespace NeuralNetwork;
int main() {
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;

    activation_functions.push_back(std::make_unique<Sigmoid>());
    activation_functions.push_back(std::make_unique<ReLu>());
    activation_functions.push_back(std::make_unique<Sigmoid>());

    Sequential sequential({{1, 2}}, std::move(activation_functions));
    Model model(std::move(sequential));
    model.print_activation_functions();


    // Выводит информацию о добавленных функциях активации
    return 0;

}