#ifndef NEURALNETWORK_SRC_MODEL_H_
#define NEURALNETWORK_SRC_MODEL_H_

#endif //NEURALNETWORK_SRC_MODEL_H_

#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include <memory>
#include "ActivationFunctions.h"
#include "LossFunction.h"
#include "DataLoader.h"
#include <fstream>

namespace NeuralNetwork {

class LinearLayer {
public:
    LinearLayer() = default;

    LinearLayer(long n, long m) {
        assert(n > 0 && m > 0);
        a = Matrix::Random(n, m);
        b = Vector::Random(n);
    }

    LinearLayer(Matrix &a_t, Vector &b_t) {
        a = a_t;
        b = b_t;
    }

    friend std::ostream &operator<<(std::ostream &os, const LinearLayer &layer) {
        os << "  A " << layer.a.rows() << " " << layer.a.cols() << "\n";
        os << layer.a << "\n";
        os << "  b " << layer.b.size() << "\n";
        os << layer.b;
        return os;
    }

    friend std::istream &operator>>(std::istream &is, LinearLayer &layer) {
        std::string naming_a;
        long n, m;
        is >> naming_a >> n >> m;
        assert(n > 0 && m > 0);
        layer.a.resize(n, m);
        for (long row = 0; row < n; row++) {
            for (long col = 0; col < m; col++) {
                is >> layer.a(row, col);
            }
        }
        std::string naming_b;
        is >> naming_b >> n;
        layer.b.resize(n);
        for (long idx = 0; idx < n; idx++) {
            is >> layer.b(idx);
        }
        return is;
    }

    Matrix a;
    Vector b;
private:
    friend class Sequential;

    [[nodiscard]] Vector Compute(const Vector &x) const {
        return a * x + b;
    }

    void Step(double learning_rate, const Matrix &da, const Vector &db) {
        a -= da * learning_rate;
        b -= db * learning_rate;
    }

};

class Sequential {
public:
    Sequential(std::initializer_list<size_t> dimensions,
               std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions_t)
        : activation_functions(std::move(activation_functions_t)) {
        assert((dimensions.size() == activation_functions.size() + 1));
        assert(!activation_functions.empty());
        assert(*dimensions.begin() == DataLoader::IMAGE_SIZE); // special for MNIST dataset
        linear_layers.reserve(activation_functions.size());
        for (auto it = dimensions.begin(); it + 1 != dimensions.end(); ++it) {
            linear_layers.emplace_back(*(it + 1), *(it));
        }
        assert(linear_layers.back().b.size() == DataLoader::COUNT_OF_DIGITS); // special for MNIST dataset

        number_of_layers = activation_functions.size();
        da.resize(number_of_layers);
        db.resize(number_of_layers);
        x_saved.resize(number_of_layers);
        y_saved.resize(number_of_layers);

        for (size_t i = 0; i < number_of_layers; ++i) {
            da[i] = linear_layers[i].a;
            db[i] = linear_layers[i].b;
        }
        Reset();
    }

    Sequential(std::vector<LinearLayer> linear_layers_t,
               std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions_t)
        : linear_layers(std::move(linear_layers_t)), activation_functions(std::move(activation_functions_t)) {
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

    void Compute(Vector &x) {
        for (size_t i = 0; i < number_of_layers; ++i) {
            x_saved[i] = x;
            y_saved[i] = activation_functions[i]->Compute(linear_layers[i].Compute(x));
            x = y_saved[i];
        }
    }

    void BackPropogate(Vector &x) {
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

    friend std::ostream &operator<<(std::ostream &os, const Sequential &sequential) {
        for (size_t i = 0; i < sequential.linear_layers.size(); i++) {
            os << i * 2 + 1 << " LinearLayer\n";
            os << sequential.linear_layers[i] << "\n";
            os << "\n" << i * 2 + 2 << " " << sequential.activation_functions[i]->GetType() << "\n";
        }
        return os;
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
        : learning_rate(learning_rate), sequential(std::move(seq)), loss_function(std::move(loss_function_t)) {

        answer.resize(DataLoader::COUNT_OF_DIGITS);
        for (int i = 0; i < DataLoader::COUNT_OF_DIGITS; ++i) {
            answer[i] = DataLoader::ConvertInt(i);
        }
    }

    void Train(DataLoader data_loader_train, DataLoader data_loader_test, size_t epoch) {
        data_loader_train.Reset();
        data_loader_test.Reset();
        Batch batch;
        for (size_t i = 0; i < epoch; ++i) {
            printf("\nepoch: %zu / %zu\n", i + 1, epoch);
            fflush(stdout);
            data_loader_train.Next(batch);
            while (!batch.empty()) {
                sequential.Reset();
                Conversion(batch);
                BackPropogate(batch);
                sequential.Step(learning_rate / static_cast<double>(data_loader_train.batch_size) * (i / 3 + 1.));
                data_loader_train.Next(batch);
            }
            data_loader_train.Reset();
            auto predict = Predict(data_loader_test);
            std::cout << static_cast<double>(predict.first) * 1.0 / static_cast<double>(predict.second) << "\n";
        }
    }

    std::pair<size_t, size_t> Predict(DataLoader &data_loader) {
        data_loader.Reset();
        size_t count_right_answers = 0;
        size_t count_all_images = 0;
        Batch batch;
        data_loader.Next(batch);
        while (!batch.empty()) {
            Conversion(batch);
            for (auto &x_y : batch) {
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

    friend std::ostream &operator<<(std::ostream &os, const Model &model) {
        os << "Model: " << model.sequential.number_of_layers * 2 << " layers " << model.loss_function->GetType()
           << " loss " << model.learning_rate
           << " lr\n";
        os << model.sequential << "\n";
        return os;
    }

    friend std::istream &operator>>(std::istream &is, Model &model) {
        double learning_rate_vol;
        size_t cnt_layer;
        std::string name_model, loss_func, loss_func_name, l_rate;
        is >> name_model >> cnt_layer >> name_model >> loss_func_name >> loss_func >> learning_rate_vol
           >> l_rate;
        cnt_layer /= 2;
        std::vector<LinearLayer> linear_layers(cnt_layer);
        std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
        activation_functions.reserve(cnt_layer);
        for (size_t i = 0; i < cnt_layer * 2; i++) {
            size_t level;
            std::string name_layer;
            is >> level >> name_layer;
            if (name_layer == "LinearLayer") {
                is >> linear_layers[i];
            } else if (name_layer == "Sigmoid") {
                activation_functions.push_back(std::make_unique<Sigmoid>());
            } else if (name_layer == "Softmax") {
                activation_functions.push_back(std::make_unique<Softmax>());
            } else if (name_layer == "ReLu") {
                activation_functions.push_back(std::make_unique<ReLu>());
            }
        }

        Sequential seq(linear_layers, std::move(activation_functions));
        model = Model(std::move(seq), std::make_unique<MSE>(), learning_rate_vol);
        return is;
    }

private:
    void Conversion(Batch &batch) {
        for (auto &x_y : batch) {
            sequential.Compute(x_y.first);
        }
    }

    void BackPropogate(const Batch &batch) {
        Vector derivative(batch[0].first.size());
        derivative.setZero();
        for (auto &x_y : batch) {
            derivative += loss_function->GetDerivative(x_y.second, x_y.first);
        }
        sequential.BackPropogate(derivative);
    }

    double learning_rate;
    Sequential sequential;
    std::unique_ptr<LossFunction> loss_function;
    std::vector<Vector> answer;
};

} // namespace NeuralNetwork
