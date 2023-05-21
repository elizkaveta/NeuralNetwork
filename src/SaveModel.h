#include <iostream>
#include <fstream>
#include <vector>
#include <initializer_list>
#include "Model.h"  // Подключите ваш заголовочный файл Model.h

namespace NeuralNetwork {

// Функция сериализации объекта Model в текстовый файл
template<typename ModelType>
void Model::SaveModel(const ModelType& model, const std::string& filename) {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + filename);
    }

    // Сохраняем значения полей в файл
    file << model.min_loss << std::endl;
    file << model.batch_size << std::endl;

    // Сохраняем activation_functions
    for (const auto& activation_function : model.activation_functions) {
        file << activation_function << std::endl;
    }

    // Сохраняем loss_function
    file << model.loss_function << std::endl;

    // Сохраняем sequential
    for (const auto& layer : model.sequential.layers) {
        file << layer.a_ << std::endl;
        file << layer.b_ << std::endl;
    }

    file.close();
}

// Функция десериализации объекта Model из текстового файла
template<typename ModelType>
void Model::LoadModel(ModelType& model, const std::string& filename) {
    std::ifstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for reading: " + filename);
    }

    // Загружаем значения полей из файла
    file >> model.min_loss;
    file >> model.batch_size;

    // Загружаем activation_functions
    typename ModelType::ActivationFunctionTemplate activation_function;
    while (file >> activation_function) {
        model.activation_functions.push_back(activation_function);
    }

    // Загружаем loss_function
    file >> model.loss_function;

    // Загружаем sequential
    typename ModelType::LinearLayer layer;
    while (file >> layer.a_ >> layer.b_) {
        model.sequential.layers.push_back(layer);
    }

    file.close();
}

}  // namespace NeuralNetwork
