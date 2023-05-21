//
// Created by Asus on 05.05.2023.
//

#ifndef NEURALNETWORKS_SRC_ACTIVATIONFUNCTION_H_
#define NEURALNETWORKS_SRC_ACTIVATIONFUNCTION_H_

#endif //NEURALNETWORKS_SRC_ACTIVATIONFUNCTION_H_
#include "../eigen/Eigen/Core"
#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include "Optimizer.h"
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class ActivationFunction {
};

class Sigmoid final : public ActivationFunction {
public:
    static Vector Compute(const Vector& x) {
        return x.array().exp() / (1 + x.array().exp());
    }
    static Matrix GetDerivative(const Vector& x) {
        return ((-x.array()).exp() / pow(1.0 + (-x.array()).exp(), 2)).matrix().asDiagonal();
    }
};

class ReLu final : public ActivationFunction {
public:
    static Vector Compute(const Vector& x) { return x.cwiseMax(0.0); }
    static Matrix GetDerivative(const Vector& x) {
        return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
    }
};
}