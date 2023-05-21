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
    Sequential(const std::initializer_list<LinearLayer>& layers) : layers(layers) {
    }
    std::initializer_list<LinearLayer> layers;
};

template<IsActivationFunction ActivationFunctionTemplate, IsLossFunction LossFunctionTemplate>
class GradientDescent {
public:
    explicit GradientDescent(double learning_rate) : learning_rate_(learning_rate) {}
    void Optimize(NeuralNetwork::LinearLayer& layer,
                  const Vector& input,
                  const Vector& expected_output,
                  const ActivationFunctionTemplate& activation_function,
                  LossFunctionTemplate loss_function) const {
        // Forward pass
        Vector output = layer.a_ * input + layer.b_;
        Vector activated_output = activation_function.ComputeLoss(output);
        // Backward pass
        Vector loss_derivative = loss_function.GetDerivative(expected_output, activated_output);
        Matrix activation_derivative = activation_function.GetDerivative(output);

        Matrix d_a = loss_derivative.asDiagonal() * activation_derivative * input.transpose();
        Vector d_b = loss_derivative.asDiagonal() * activation_derivative;

        // Update weights and biases
        layer.a_ -= learning_rate_ * d_a;
        layer.b_ -= learning_rate_ * d_b;
    }

private:
    double learning_rate_;
};

template<IsLossFunction LossFunctionTemplate>
class Model {
public:
    Model(const Sequential& sequential,
          std::vector<std::unique_ptr<ActivationFunction> >& activation_functions_t,
          const LossFunctionTemplate& loss_function_t, double min_loss, GradientDescent<ActivationFunctionTemplate, LossFunctionTemplate> d = GradientDescent<ActivationFunctionTemplate, LossFunctionTemplate>(),
           size_t b_size = 32)
        : sequential(sequential), min_loss(min_loss) {
        activation_functions = std::move(activation_functions_t);
        loss_function = loss_function_t;
        batch_size = b_size;
    }
    void Train(size_t epoch) {

    }

private:
    std::vector<std::unique_ptr<ActivationFunction> > activation_functions;
    LossFunctionTemplate loss_function;
    Sequential sequential;
    double min_loss;
    size_t batch_size;
    std::initializer_list<Vector> results;
};

}

#endif //NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
