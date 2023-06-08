#include "Sequential.h"

namespace NeuralNetwork {
    LinearLayer::LinearLayer(long n, long m) {
        assert(n > 0 && m > 0);
        a = Matrix::Random(n, m);
        b = Vector::Random(n);
    }

    LinearLayer::LinearLayer(Matrix &a_t, Vector &b_t) {
        a = a_t;
        b = b_t;
    }

    [[nodiscard]] Vector LinearLayer::Compute(const Vector &x) const {
        return a * x + b;
    }

    void LinearLayer::Step(double learning_rate, const Matrix &da, const Vector &db) {
        a -= da * learning_rate;
        b -= db * learning_rate;
    }

    Sequential::Sequential(std::initializer_list<size_t> dimensions,
               std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions_t)
        : activation_functions(std::move(activation_functions_t)) {
        assert((dimensions.size() == activation_functions.size() + 1));
        assert(!activation_functions.empty());
        assert(*dimensions.begin() == MNIST::IMAGE_SIZE); // special for MNIST dataset
        linear_layers.reserve(activation_functions.size());
        for (auto it = dimensions.begin(); it + 1 != dimensions.end(); ++it) {
            linear_layers.emplace_back(*(it + 1), *(it));
        }
        assert(linear_layers.back().b.size() == MNIST::COUNT_OF_DIGITS); // special for MNIST dataset

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

    Sequential::Sequential(std::vector<LinearLayer> linear_layers_t,
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

    void Sequential::Compute(Vector &x) {
        for (size_t i = 0; i < number_of_layers; ++i) {
            x_saved[i] = x;
            y_saved[i] = activation_functions[i]->Compute(linear_layers[i].Compute(x));
            x = y_saved[i];
        }
    }

    void Sequential::BackPropogate(Vector &x) {
        for (int32_t i = static_cast<int>(number_of_layers) - 1; i >= 0; --i) {
            Vector activ_x = activation_functions[i]->GetDerivative(y_saved[i]) * x;
            da[i] += activ_x * x_saved[i].transpose();
            db[i] += activ_x;
            x = linear_layers[i].a.transpose() * activ_x;
        }
    }

    void Sequential::Step(double learning_rate) {
        for (size_t i = 0; i < number_of_layers; ++i) {
            linear_layers[i].Step(learning_rate, da[i], db[i]);
        }
    }

    void Sequential::Reset() {
        for (size_t i = 0; i < number_of_layers; ++i) {
            da[i].setZero();
            db[i].setZero();
        }
    }

}

