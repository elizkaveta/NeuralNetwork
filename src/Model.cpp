#include "Model.h"

namespace NeuralNetwork {

    Model::Model(Sequential seq, std::unique_ptr<LossFunction> loss_function_t, double learning_rate)
    : learning_rate(learning_rate), sequential(std::move(seq)), loss_function(std::move(loss_function_t)) {

        answer.resize(MNIST::COUNT_OF_DIGITS);
        for (int32_t i = 0; i < MNIST::COUNT_OF_DIGITS; ++i) {
            answer[i] = MNIST::ConvertInt(i);
        }
    }

    void Model::Train(const std::string& path_to_images, const std::string& path_to_lables, size_t batch_size, size_t epoch) {
        DataLoader data_loader_train(path_to_images, path_to_lables, batch_size);
        Batch batch;
        for (size_t i = 0; i < epoch; ++i) {
            data_loader_train.Next(batch);
            while (!batch.empty()) {
                BackForwardPropogate(batch, learning_rate / (data_loader_train.batch_size* (i / 3 + 1.)));
                data_loader_train.Next(batch);
            }
            data_loader_train.Reset();
        }
    }

    std::pair<size_t, size_t> Model::Predict(const std::string& path_to_images, const std::string& path_to_lables, size_t batch_size) {
        DataLoader data_loader_test(path_to_images, path_to_lables, batch_size);
        data_loader_test.Reset();
        size_t count_right_answers = 0;
        size_t count_all_images = 0;
        Batch batch;
        data_loader_test.Next(batch);
        while (!batch.empty()) {
            Conversion(batch);
            for (auto &x_y : batch) {
                if (MNIST::ConvertVector(x_y.second) == MNIST::ConvertVector(x_y.first)) {
                    ++count_right_answers;
                }
            }
            count_all_images += batch.size();
            data_loader_test.Next(batch);
        }
        data_loader_test.Reset();
        return std::make_pair(count_right_answers, count_all_images);
    }

    void Model::BackForwardPropogate(Batch& batch, double learning_rate_epoch) {
        sequential.Reset();
        Conversion(batch);
        BackPropogate(batch);
        sequential.Step(learning_rate_epoch);
    }

    void Model::Conversion(Batch &batch) {
        for (auto &x_y : batch) {
            sequential.Compute(x_y.first);
        }
    }

    void Model::BackPropogate(const Batch &batch) {
        Vector derivative(batch[0].first.size());
        derivative.setZero();
        for (auto &x_y : batch) {
            derivative += loss_function->GetDerivative(x_y.second, x_y.first);
        }
        sequential.BackPropogate(derivative);
    }
}