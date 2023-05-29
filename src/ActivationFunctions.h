#ifndef NEURALNETWORK_SRC_ACTIVATIONFUNCTION_H_
#define NEURALNETWORK_SRC_ACTIVATIONFUNCTION_H_

#endif //NEURALNETWORK_SRC_ACTIVATIONFUNCTION_H_

#include <utility>
#include <vector>
#include <iostream>
#include <string>
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class ActivationFunction {
 public:
  virtual Vector Compute(const Vector &x) = 0;
  virtual Matrix GetDerivative(const Vector &x) = 0;
  virtual std::string GetType() = 0;
};

class Sigmoid final : public ActivationFunction {
 public:
  Vector Compute(const Vector &x) override {
    return 1 / (1 + (-x.array()).exp());
  }
  Matrix GetDerivative(const Vector &x) override {
    return ((-x.array()).exp() / pow(1.0 + (-x.array()).exp(), 2)).matrix().asDiagonal();
  }
  std::string GetType() override {
    return "Sigmoid";
  }
};

class ReLu final : public ActivationFunction {
 public:
  Vector Compute(const Vector &x) override {
    return x.cwiseMax(0.0);
  }
  Matrix GetDerivative(const Vector &x) override {
    return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
  }
  std::string GetType() override {
    return "ReLu";
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
  std::string GetType() override {
    return "Softmax";
  }
};

}// namespace NeuralNetwork
