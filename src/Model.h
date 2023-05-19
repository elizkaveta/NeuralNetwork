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

class Sequential {
public:
    explicit Sequential(const std::vector<LinearLayer>& layers) : layers(layers) {
    }

    std::vector<LinearLayer> layers;
};

template <typename T>
concept IsActivationFunction = requires(T a, Vector b, Matrix c) {
    { a.Compute(b) } -> std::same_as<Vector>;
    { a.GetDerivative(b) } -> std::same_as<Matrix>;
};

template <typename T>
concept IsLossFunction = requires(T a, Vector y1, Vector y2) {
    { a.ComputeLoss(y1, y2) } -> std::same_as<double>;
    { a.GetDerivative(y1, y2) } -> std::same_as<Vector>;
};


template <IsActivationFunction ActivationFunctionTemplate, IsLossFunction LossFunctionTemplate>
class Model {
public:
    Model(Sequential sequential, const ActivationFunctionTemplate& activation_function1, const LossFunctionTemplate& loss_function1, double min_loss)
        : sequential(std::move(sequential)), min_loss(min_loss) {
        activation_function = std::make_unique<ActivationFunctionTemplate>(std::move(activation_function1));
        loss_function = std::make_unique<LossFunctionTemplate>(std::move(loss_function1));
    }
    void Train() {

    }
private:
    std::unique_ptr<ActivationFunctionTemplate> activation_function;
    std::unique_ptr<LossFunctionTemplate> loss_function;
    Sequential sequential;
    double min_loss;
};
}

#endif //NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
