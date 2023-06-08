#include "LossFunction.h"

namespace NeuralNetwork {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

[[nodiscard]] double MSE::Compute(const Vector &expected,
                             const Vector &predicted) const {
    return (expected - predicted).squaredNorm() / expected.size();
}

[[nodiscard]] Vector MSE::GetDerivative(const Vector &expected,
                                   const Vector &predicted) const {
    return 2 * (predicted - expected);
}

[[nodiscard]] std::string MSE::GetType() const {
    return "MSE";
}
}
