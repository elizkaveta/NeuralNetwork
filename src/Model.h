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
    LinearLayer(size_t n, size_t m) {
        a = Matrix::Random(n, m);
        b = Vector::Random(n);
    }
    Matrix a;
    Vector b;
private:
    friend class Sequential;
    [[nodiscard]] Vector Compute(const Vector& x) const {
        return a * x + b;
    }
    void Step(double learning_rate, const Matrix& da, const Vector& db) {
        a -= da * learning_rate;
        b -= db * learning_rate;
    }
};

class Sequential {
public:
    Sequential(std::initializer_list<size_t> dimensions,
               std::vector<std::unique_ptr<ActivationFunction>> activation_functions_t)
        : activation_functions(std::move(activation_functions_t)) {
        assert((dimensions.size() == activation_functions.size() + 1));
        assert(!activation_functions.empty());
        assert(*dimensions.begin() == DataLoader::IMAGE_SIZE);
        linear_layers.reserve(activation_functions.size());
        for (auto it = dimensions.begin(); it + 1 != dimensions.end(); ++it) {
            linear_layers.emplace_back(*(it + 1), *(it));
        }
        number_of_layers = activation_functions.size();
        da.resize(number_of_layers);
        db.resize(number_of_layers);
        for (size_t i = 0; i < number_of_layers; ++i) {
            da[i] = linear_layers[i].a;
            db[i] = linear_layers[i].b;
        }
        Reset();
        x_saved.resize(number_of_layers);
        y_saved.resize(number_of_layers);
    }
private:
    friend class Model;
    Vector Compute(const Vector& x, size_t i) {
        x_saved[i] = x;
        y_saved[i] = activation_functions[i]->Compute(linear_layers[i].Compute(x));
        return y_saved[i];
    }

    void BackPropogate(Vector& x) {
        for (int i = static_cast<int>(number_of_layers) - 1; i >= 0; --i) {
            Vector activ_x = activation_functions[i]->GetDerivative(y_saved[i]) * x;
            da[i] += activ_x * x_saved[i].transpose();
            db[i] += activ_x;
            x = linear_layers[i].a.transpose() * activ_x;
        }
    }

    void Step(double learning_rate) {
        for (size_t i = 0; i < number_of_layers; ++i) {
            linear_layers[i].Step(learning_rate, da[i], db[i]);
        }
    }

    void Reset() {
        for (size_t i = 0; i < number_of_layers; ++i) {
            da[i].setZero();
            db[i].setZero();
        }
    }

    size_t number_of_layers = 0;
    std::vector<Matrix> da;
    std::vector<Vector> db;
    std::vector<Vector> x_saved;
    std::vector<Vector> y_saved;
    std::vector<LinearLayer> linear_layers;
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
};

class Model {
public:
    Model(Sequential seq, std::unique_ptr<LossFunction> loss_function_t, double learning_rate)
        : sequential(std::move(seq)), loss_function(std::move(loss_function_t)), learning_rate(learning_rate) {

        answer.resize(DataLoader:: COUNT_OF_DIGITS);
        for (int i = 0; i < DataLoader::COUNT_OF_DIGITS; ++i) {
            answer[i] = DataLoader::ConvertInt(i);
        }
    }

    void Train(DataLoader& data_loader, DataLoader& data_loader_test, size_t epoch = 10) {
        data_loader.Reset();
        data_loader_test.Reset();
        Batch batch;
        for (size_t i = 0; i < epoch; ++i) {
            printf("\nepoch: %zu / %zu\n", i + 1, epoch);
            fflush(stdout);
            data_loader.Next(batch);
            while (!batch.empty()) {
                sequential.Reset();
                Conversion(batch);
                BackPropogate(batch);
                sequential.Step(learning_rate / (data_loader.batch_size * (i / 3 + 1.)) );
                data_loader.Next(batch);
            }
            data_loader.Reset();
            auto predict = Predict(data_loader_test);
            printf("accuracy test: %f\n", predict.first * 1.0/predict.second);
        }
    }

    std::pair<size_t, size_t> Predict(DataLoader& data_loader) {
        data_loader.Reset();
        size_t count_right_answers = 0;
        size_t count_all_images = 0;
        Batch batch;
        data_loader.Next(batch);
        while (!batch.empty()) {
            Conversion(batch);
            for (auto& x_y : batch) {
                if (DataLoader::ConvertVector(x_y.second) == DataLoader::ConvertVector(x_y.first)) {
                    ++count_right_answers;
                }
            }
            count_all_images += batch.size();
            data_loader.Next(batch);
        }
        data_loader.Reset();
        return std::make_pair(count_right_answers, count_all_images);
    }

private:
    void Conversion(Batch& batch) {
        for (auto& x_y : batch) {
            for (size_t i = 0; i < sequential.number_of_layers; ++i) {
                x_y.first = sequential.Compute(x_y.first, i);
            }
        }
    }

    void BackPropogate(const Batch& batch) {
        Vector derivative(batch[0].first.size());
        derivative.setZero();
        for (auto& x_y : batch) {
            derivative += loss_function->GetDerivative(x_y.second, x_y.first);
        }
        sequential.BackPropogate(derivative);
    }

    double learning_rate;
    Sequential sequential;
    std::unique_ptr<LossFunction> loss_function;
    std::vector<Vector> answer;
};

}

#endif //NEURAL_NETWORKS_FROM_SCRATCH_ON_C_SRC_MATRIXES_MATRIX_H_
