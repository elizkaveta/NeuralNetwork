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
public:
    virtual Vector Compute(const Vector &x) = 0;
    virtual Matrix GetDerivative(const Vector &x) = 0;
};

class Sigmoid final : public ActivationFunction {
public:
    Vector Compute(const Vector &x) override {
        return 1 / (1 + (-x.array()).exp());
    }
    Matrix GetDerivative(const Vector &x) override {
        return ((-x.array()).exp() / pow(1.0 + (-x.array()).exp(), 2)).matrix().asDiagonal();
    }
};

class ReLu final : public ActivationFunction {
public:
    Vector Compute(const Vector &x) override {
        return x.cwiseMax(0.0);
    }
    Matrix GetDerivative(const Vector &x) override {
        return (x.array() > 0).cast<double>().matrix().asDiagonal();
    }
};

class Softmax : public ActivationFunction {
public:
    Vector Compute(const Vector &x) override {
        return (x.array().exp() / x.array().exp().sum()).matrix();
    }

    Matrix GetDerivative(const Vector &x) override {
        size_t n = x.rows();
        Matrix ans(n, n);
        Vector compute = Compute(x);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = 0; j < n; ++j) {
                if (i == j) {
                    ans(i, j) = compute(i) * (1 - compute(j));
                } else {
                    ans(i, j) = -compute(i) * compute(j);
                }
            }
        }

        return ans;
    }
};

}