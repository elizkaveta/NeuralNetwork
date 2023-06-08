#ifndef NEURALNETWORK_SRC_SEQUENTIAL_H_
#define NEURALNETWORK_SRC_SEQUENTIAL_H_

#endif //NEURALNETWORK_SRC_SEQUENTIAL_H_
#include <vector>
#include "../eigen/Eigen/Dense"
#include "MNIST.h"
#include "ActivationFunctions.h"

namespace NeuralNetwork {

using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;

struct LinearLayer {
public:
    LinearLayer() = default;

    LinearLayer(long n, long m);

    LinearLayer(Matrix &a_t, Vector &b_t);

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

    [[nodiscard]] Vector Compute(const Vector &x) const;

    friend class Sequential;
private:
    void Step(double learning_rate, const Matrix &da, const Vector &db);


    Matrix a;
    Vector b;
};

class Sequential {
public:
    Sequential(std::initializer_list<size_t> dimensions,
               std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions_t);

    Sequential(std::vector<LinearLayer> linear_layers_t,
               std::vector<std::unique_ptr<ActivationFunction>> &&activation_functions_t);

private:
    friend class Model;

    void Compute(Vector &x);

    void BackPropogate(Vector &x);

    void Step(double learning_rate);

    void Reset();

    friend std::ostream &operator<<(std::ostream &os, const Sequential &sequential) {
        for (size_t i = 0; i < sequential.linear_layers.size(); i++) {
            os << i * 2 + 1 << " LinearLayer\n";
            os << sequential.linear_layers[i] << "\n";
            os << "\n" << i * 2 + 2 << " " << sequential.activation_functions[i]->GetType() << "\n";
        }
        return os;
    }

    friend std::istream &operator>>(std::istream &is, Sequential &seq) {
        size_t cnt_layer;
        is >> cnt_layer;
        cnt_layer /= 2;
        std::vector<LinearLayer> linear_layers_t(cnt_layer);
        std::vector<std::unique_ptr<ActivationFunction>> activation_functions_t;
        activation_functions_t.reserve(cnt_layer);
        for (size_t i = 0; i < cnt_layer * 2; i++) {
            size_t level;
            std::string name_layer;
            is >> level >> name_layer;
            if (name_layer == "LinearLayer") {
                is >> linear_layers_t[i];
            } else if (name_layer == "Sigmoid") {
                activation_functions_t.push_back(std::make_unique<Sigmoid>());
            } else if (name_layer == "Softmax") {
                activation_functions_t.push_back(std::make_unique<Softmax>());
            } else if (name_layer == "ReLu") {
                activation_functions_t.push_back(std::make_unique<ReLu>());
            }
        }
        seq = Sequential(linear_layers_t, std::move(activation_functions_t));
        return is;
    }

    size_t number_of_layers = 0;
    std::vector<Matrix> da;
    std::vector<Vector> db;
    std::vector<Vector> x_saved;
    std::vector<Vector> y_saved;
    std::vector<LinearLayer> linear_layers;
    std::vector<std::unique_ptr<ActivationFunction>> activation_functions;
};

}