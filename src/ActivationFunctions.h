//
// Created by Asus on 05.05.2023.
//

#ifndef NEURALNETWORKS_SRC_ACTIVATIONFUNCTION_H_
#define NEURALNETWORKS_SRC_ACTIVATIONFUNCTION_H_

#endif //NEURALNETWORKS_SRC_ACTIVATIONFUNCTION_H_
#include <utility>
#include <vector>
#include <iostream>
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class ActivationFunction {
    virtual Vector Compute(const Vector &x) = 0;
    virtual Matrix GetDerivative(const Vector &x) = 0;
    friend class Sequential;
};

class Sigmoid final : public ActivationFunction {
    Vector Compute(const Vector &x) override {
        return 1 / (1 + (-x.array()).exp());
    }
    Matrix GetDerivative(const Vector &x) override {
        return ((-x.array()).exp() / pow(1.0 + (-x.array()).exp(), 2)).matrix().asDiagonal();
    }
};

class ReLu final : public ActivationFunction {
    Vector Compute(const Vector &x) override {
        return x.cwiseMax(0.0);
    }
    Matrix GetDerivative(const Vector &x) override {
        return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
    }
};

class Softmax : public ActivationFunction {
public:
    Vector Compute(const Vector &x) override {
        auto result = x.array().exp();
        return result / result.sum();
    }

    Matrix GetDerivative(const Vector &x) override {
        Vector computeSoftmax = Compute(x);
        Matrix diagonal = computeSoftmax.asDiagonal();
        return diagonal - computeSoftmax * computeSoftmax.transpose();
    }
};

}