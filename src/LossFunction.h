//
// Created by Asus on 05.05.2023.
//

#ifndef NEURALNETWORKS_SRC_LOSSFUNCTION_H_
#define NEURALNETWORKS_SRC_LOSSFUNCTION_H_

#endif //NEURALNETWORKS_SRC_LOSSFUNCTION_H_
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

class LossFunction {
public:
    virtual double Compute(const Vector& expected_y,
                               const Vector& predicted_y) = 0;
    virtual Vector GetDerivative(const Vector& expected_y,
                                 const Vector& predicted_y) = 0;
};

class MSE : public LossFunction {
public:
    double Compute(const Vector& y_e,
                       const Vector& y_p) final {
        return (y_e - y_p).squaredNorm() / y_e.size();
    }
    Vector GetDerivative(const Vector& y_e,
                         const Vector& y_p) final {
        return 2 * (y_p - y_e);
    }
};
}