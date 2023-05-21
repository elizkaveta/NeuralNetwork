#ifndef NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
#define NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_

#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include <memory.h>
#include "Optimizer.h"
#include "ActivationFunctions.h"
#include "LossFunction.h"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class DataLoader {
public:
    DataLoader() = default;

    std::vector<Vector> Next(int n) {
        return {};
    }
};

class LinearLayer {
public:
    Matrix a_;
    Vector b_;
    LinearLayer(size_t n, size_t m) : a_(Matrix::Random(m, n)), b_(Vector::Random(m)) {
    }
};

template<typename T>
concept IsActivationFunction = requires(T a, Vector b, Matrix c) {
    { a.Compute(b) } -> std::same_as<Vector>;
    { a.GetDerivative(b) } -> std::same_as<Matrix>;
};

template<typename T>
concept IsLossFunction = requires(T a, Vector y1, Vector y2) {
    { a.ComputeLoss(y1, y2) } -> std::same_as<double>;
    { a.GetDerivative(y1, y2) } -> std::same_as<Vector>;
};

class Sequential {
public:
    Sequential(std::initializer_list<LinearLayer> linear_layers,
               std::vector<std::unique_ptr<ActivationFunction>> activation_functions)
        : linear_layers(linear_layers), activation_functions(std::move(activation_functions)) {}

    void print_activation_functions() const {
        for (size_t i = 0; i < activation_functions.size(); i++) {
            std::cout << "Function " << i << " type: " << typeid(*activation_functions[i]).name() << std::endl;
        }
    }

private:
    std::vector<LinearLayer> linear_layers;
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
};

class Model {
public:
    Model(Sequential seq) : sequential_(std::move(seq)) {}

    void print_activation_functions() const {
        sequential_.print_activation_functions();
    }

private:
    Sequential sequential_;
};

}

#endif //NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
