#ifndef NEURALNETWORK_SRC_LOSSFUNCTION_H_
#define NEURALNETWORK_SRC_LOSSFUNCTION_H_

#endif //NEURALNETWORK_SRC_LOSSFUNCTION_H_

#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class LossFunction {
 private:
  friend class Sequential;
  friend class Model;
  virtual double Compute(const Vector &expected,
                         const Vector &predicted) = 0;
  virtual Vector GetDerivative(const Vector &expected,
                               const Vector &predicted) = 0;
};

class MSE : public LossFunction {
 private:
  double Compute(const Vector &expected,
                 const Vector &predicted) final {
    return (expected - predicted).squaredNorm() / expected.size();
  }
  Vector GetDerivative(const Vector &expected,
                       const Vector &predicted) final {
    return 2 * (predicted - expected);
  }
};

} // namespace NeuralNetwork