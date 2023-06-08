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
    friend class Sequential;
    friend class Model;
private:
    [[nodiscard]] virtual Vector Compute(const Vector &x) const = 0;

    [[nodiscard]] virtual Matrix GetDerivative(const Vector &x) const = 0;

    [[nodiscard]] virtual std::string GetType() const = 0;
};

class Sigmoid final : public ActivationFunction {
private:
    [[nodiscard]] Vector Compute(const Vector &x) const final;

    [[nodiscard]] Matrix GetDerivative(const Vector &x) const final;

    [[nodiscard]] std::string GetType() const final;
};

class ReLu final : public ActivationFunction {
private:
    [[nodiscard]] Vector Compute(const Vector &x) const final {
        return x.cwiseMax(0.0);
    }

    [[nodiscard]] Matrix GetDerivative(const Vector &x) const final {
        return (x.array() > 0.0).cast<double>().matrix().asDiagonal();
    }

    [[nodiscard]] std::string GetType() const final {
        return "ReLu";
    }
};

class Softmax : public ActivationFunction {
private:
    [[nodiscard]] Vector Compute(const Vector &x) const final {
        auto result = x.array().exp();
        return result / result.sum();
    }

    [[nodiscard]] Matrix GetDerivative(const Vector &x) const final {
        Vector computeSoftmax = Compute(x);
        Matrix diagonal = computeSoftmax.asDiagonal();
        return diagonal - computeSoftmax * computeSoftmax.transpose();
    }

    [[nodiscard]] std::string GetType() const final {
        return "Softmax";
    }
};

}// namespace NeuralNetwork
