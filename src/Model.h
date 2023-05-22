#ifndef NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
#define NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_

#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include <memory.h>
#include "Optimizer.h"
#include "ActivationFunctions.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include <fstream>

namespace NeuralNetwork {

class LinearLayer {
public:
    Matrix a;
    Vector b;
    LinearLayer(size_t n, size_t m) : a(Matrix::Random(m, n)), b(Vector::Random(m)) {
    }
    Vector Compute(const Vector& x) {
        return a * x + b;
    }
    void Step(double learning_rate, const Matrix& da, const Vector& db, size_t batch_size) {
        a -= learning_rate * da / batch_size;
        b -= learning_rate * db / batch_size;
    }
};

class Sequential {
public:
    Sequential(std::initializer_list<LinearLayer> linear_layers_t,
               std::vector<std::unique_ptr<ActivationFunction>> activation_functions_t)
        : linear_layers(linear_layers_t), activation_functions(std::move(activation_functions_t)) {
        assert((linear_layers.size() == activation_functions.size()));
        number_of_layers = linear_layers.size();
        da.resize(number_of_layers);
        db.resize(number_of_layers);
        for (size_t i = 0; i < number_of_layers; ++i) {
            da[i] = linear_layers[i].a;
            da[i].setZero();
            db[i] = linear_layers[i].b;
            db[i].setZero();
        }
    }

    Vector Compute(const Vector& x, size_t i) {
        x_[i] = x.transpose(); //.t
        y_[i] = activation_functions[i]->Compute(linear_layers[i].Compute(x));
        return y_[i];
    }
    Vector BackPropogate(const Vector& x, size_t i) {
        Vector activ_x = activation_functions[i]->GetDerivative(y_[i]) * x;
        da[i] += activ_x * x_[i];
        db[i] += activ_x * x;
        return linear_layers[i].a.transpose() * activ_x;
    }
    void Step(double learning_rate, size_t batch_size, size_t i) {
        linear_layers[i].Step(learning_rate, da[i], db[i], batch_size);
    }
    void Reset(size_t i) {
        da[i].setZero();
        db[i].setZero();
    }
    std::vector<Matrix> da;
    std::vector<Vector> db;
    std::vector<Vector> x_;
    std::vector<Vector> y_;
    std::vector<LinearLayer> linear_layers;
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
    size_t number_of_layers = 0;
};

class Model {
public:
    Model(Sequential seq, std::unique_ptr<LossFunction> loss_function_t, double learning_rate)
        : sequential(std::move(seq)), loss_function(std::move(loss_function_t)), learning_rate(learning_rate) {
        answer.resize(10);
        for (int i = 0; i < 10; ++i) {
            answer[i] = DataLoader::ConvertInt(i);
        }
    }
    void Train(DataLoader& data_loader, size_t epoch = 20) {

        for (size_t i = 0; i < epoch; ++i) {
            printf("hui");
            Batch batch;
            Reset();
            while (!(batch = data_loader.Next()).empty()) {
                Batch pred_y = Conversion(batch);
                BackPropogate(pred_y);
            }
            Step(learning_rate, batch.size());
        }
    }
    int Predict(Batch batch) {
        batch = Conversion(batch);
        int count_right = 0;
        for (auto& x_y : batch) {
            int ans = 0;
            double loss_min = 1e9;
            for (int i = 0; i < 10; ++i) {
                double loss = loss_function->Compute(x_y.second, x_y.first);
                if (loss < loss_min) {
                    ans = i;
                    loss = loss_min;
                }
            }
            if (DataLoader::ConvertVector(x_y.second) == ans) {
                ++count_right;
            }
        }
        return count_right;
    }
private:
    Batch Conversion(Batch batch) {
        for (auto& [x, y] : batch) {
            for (size_t i = 0; i < sequential.number_of_layers; ++i) {
                x = sequential.Compute(x, i);
            }
        }
        return batch;
    }
    void BackPropogate(Batch batch) {
        Vector derivative;
        for (auto& x_y : batch) {
            derivative += loss_function->GetDerivative(x_y.first, x_y.second);
        }
        for (size_t i = sequential.number_of_layers - 1; i >= 0; --i) {
            derivative = sequential.BackPropogate(derivative, i);
        }
    }

    void Step(double lr, size_t batch_size) {
        for (size_t i = 0; i < sequential.number_of_layers; ++i) {
            sequential.Step(lr, batch_size, i);
        }
    }

    void Reset() {
        for (size_t i = 0; i < sequential.number_of_layers; ++i) {
            sequential.Reset(i);
        }
    }
    double learning_rate;
    Sequential sequential;
    std::unique_ptr<LossFunction> loss_function;
    std::vector<Vector> answer;
};

}

#endif //NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
