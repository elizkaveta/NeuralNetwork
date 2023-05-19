#ifndef NEURAL_NETWORKS_FROM_SCRATCH_ON_C__SRC_OPTIMIZER_H_
#define NEURAL_NETWORKS_FROM_SCRATCH_ON_C__SRC_OPTIMIZER_H_
#include "../eigen/Eigen/Dense"

namespace NeuralNetwork {
class Optimizer {
    virtual void dest() = 0;
};

class GradientDescentNormal : public Optimizer {
    virtual void dest() {

    }
};

class ADAM : public Optimizer {
    virtual void dest() {

    }
};

class SAG : public Optimizer {
    virtual void dest() {

    }
};
}
#endif //NEURAL_NETWORKS_FROM_SCRATCH_ON_C__SRC_OPTIMIZER_H_
