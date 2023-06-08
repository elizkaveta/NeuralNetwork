#ifndef NEURALNETWORK_SRC_MODEL_H_
#define NEURALNETWORK_SRC_MODEL_H_

#endif //NEURALNETWORK_SRC_MODEL_H_

#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include "LossFunction.h"
#include "DataLoader.h"
#include "Sequential.h"

namespace NeuralNetwork {

class Model {
public:
    Model(Sequential seq, std::unique_ptr<LossFunction> loss_function_t, double learning_rate);

    void Train(const std::string& path_to_images, const std::string& path_to_lables, size_t batch_size, size_t epoch);

    std::pair<size_t, size_t> Predict(const std::string& path_to_images, const std::string& path_to_lables, size_t batch_size);

    friend std::ostream &operator<<(std::ostream &os, const Model &model) {
        os << "Model: " << " layers " << model.loss_function->GetType()
           << " loss " << model.learning_rate
           << " lr " << model.sequential.number_of_layers * 2 << "\n";
        os << model.sequential << "\n";
        return os;
    }

    friend std::istream &operator>>(std::istream &is, Model &model) {
        double learning_rate_vol;
        Sequential seq({784, 10}, {});
        std::string name_model, loss_func, loss_func_name, l_rate;
        is >> name_model >> name_model >> loss_func_name >> loss_func >> learning_rate_vol
           >> l_rate >> seq;
        model = Model(std::move(seq), std::make_unique<MSE>(), learning_rate_vol);
        return is;
    }

private:
    void BackForwardPropogate(Batch& batch, double learning_rate_epoch);

    void Conversion(Batch &batch);

    void BackPropogate(const Batch &batch);

    double learning_rate;
    Sequential sequential;
    std::unique_ptr<LossFunction> loss_function;
    std::vector<Vector> answer;
};

} // namespace NeuralNetwork
