#include "ActivationFunctions.h"

namespace NeuralNetwork {
    [[nodiscard]] Vector Sigmoid::Compute(const Vector &x) const {
        return 1 / (1 + (-x.array()).exp());
    }
    [[nodiscard]] Matrix Sigmoid::GetDerivative(const Vector &x) const {
        return ((-x.array()).exp() / pow(1.0 + (-x.array()).exp(), 2)).matrix().asDiagonal();
    }
    [[nodiscard]] std::string Sigmoid::GetType() const {
        return "Sigmoid";
    }
} // end namespace NeuralNetwork
