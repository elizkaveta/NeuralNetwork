#ifndef NEURALNETWORK_SRC_LOSSFUNCTION_H_
#define NEURALNETWORK_SRC_LOSSFUNCTION_H_

#endif //NEURALNETWORK_SRC_LOSSFUNCTION_H_
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class LossFunction {
public:
    friend class Model;
private:
    [[nodiscard]] virtual double Compute(const Vector &expected,
                           const Vector &predicted) const = 0;

    [[nodiscard]] virtual Vector GetDerivative(const Vector &expected,
                                 const Vector &predicted) const = 0;

    [[nodiscard]] virtual std::string GetType() const = 0;
};

class MSE final : public LossFunction {
private:
    [[nodiscard]] double Compute(const Vector &expected,
                   const Vector &predicted) const final;

    [[nodiscard]] Vector GetDerivative(const Vector &expected,
                         const Vector &predicted) const final;

    [[nodiscard]] std::string GetType() const final;
};

} // namespace NeuralNetwork
