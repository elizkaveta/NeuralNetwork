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
    LinearLayer(size_t n, size_t m) {
        a = Matrix::Random(n, m);
        b = Vector::Random(n);
    }
    Vector Compute(const Vector& x) const {
        return a * x + b;
    }
    void Step(double learning_rate, const Matrix& da, const Vector& db, size_t batch_size) {
        a -= learning_rate * da / batch_size;
        b -= learning_rate * db / batch_size;
    }
};

class Sequential {
public:
    Sequential(std::initializer_list<size_t> dimensions,
               std::vector<std::unique_ptr<ActivationFunction>> activation_functions_t)
        : activation_functions(std::move(activation_functions_t)) {
        assert((dimensions.size() == activation_functions.size() + 1));

        linear_layers.reserve(activation_functions.size());

        for (auto it = dimensions.begin(); it + 1 != dimensions.end(); ++it) {
            linear_layers.emplace_back(*(it + 1), *(it));
        }
        number_of_layers = activation_functions.size();
        da.resize(number_of_layers);
        db.resize(number_of_layers);
        for (size_t i = 0; i < number_of_layers; ++i) {
            da[i] = linear_layers[i].a;
            da[i].setZero();
            db[i] = linear_layers[i].b;
            db[i].setZero();
        }
        x_.resize(number_of_layers);
        y_.resize(number_of_layers);
    }

    Vector Compute(const Vector& x, size_t i) {
        x_[i] = x;
        y_[i] = activation_functions[i]->Compute(linear_layers[i].Compute(x));
        return y_[i];
    }
    Vector BackPropogate(const Vector& x, size_t i) {
        Vector activ_x = activation_functions[i]->GetDerivative(y_[i]) * x;
        da[i] += activ_x * x_[i].transpose();
        db[i] += activ_x;
        return linear_layers[i].a.transpose() * activ_x;
    }
    void Step(double learning_rate, size_t batch_size, size_t i) {
        linear_layers[i].Step(learning_rate, da[i], db[i], batch_size);
    }
    void Reset(size_t i) {
        da[i].setZero();
        db[i].setZero();
    }
    size_t number_of_layers = 0;
private:
    std::vector<Matrix> da;
    std::vector<Vector> db;
    std::vector<Vector> x_;
    std::vector<Vector> y_;
    std::vector<LinearLayer> linear_layers;
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
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

    void Train(DataLoader& data_loader, size_t epoch = 3) {
        for (size_t i = 0; i < epoch; ++i) {
            printf("epoch: %zu / %zu\n", i + 1, epoch);
            fflush(stdout);
            Batch batch = data_loader.Next();
            while (!batch.empty()) {
                Reset();
                Conversion(batch);
                BackPropogate(batch);
                batch = data_loader.Next();
                Step(learning_rate / (i + 1), data_loader.batch_size);
            }
            data_loader.Reset();
        }
    }

    int Predict(Batch batch, std::vector<int>& a) {
        Conversion(batch);
        int count_right = 0;
        for (auto& x_y : batch) {
            int ans = 0;
            double similar_max = -1e9;
            for (int i = 0; i < 10; ++i) {
                if (x_y.first[i] > similar_max) {
                    ans = i;
                    similar_max = x_y.first[i];
                }
            }
            a[ans]++;
            if (DataLoader::ConvertVector(x_y.second) == ans) {

                ++count_right;
            }
        }
        return count_right;
    }

private:
    void Conversion(Batch& batch) {
        for (size_t j = 0; j < batch.size(); ++j) {
            for (size_t i = 0; i < sequential.number_of_layers; ++i) {
                batch[j].first = sequential.Compute(batch[j].first, i);
            }
        }
    }

    void BackPropogate(Batch batch) {
        Vector derivative(batch[0].first.size());
        for (auto& x_y : batch) {
            derivative += loss_function->GetDerivative(x_y.first, x_y.second);
        }
        for (int i = static_cast<int>(sequential.number_of_layers) - 1; i >= 0; --i) {
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
