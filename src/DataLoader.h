//
// Created by Elizaveta Iashchinskaia on 22.05.2023.
//

#ifndef NEURALNETWORK_SRC_DATALOADER_H_
#define NEURALNETWORK_SRC_DATALOADER_H_

#include "../eigen/Eigen/Dense"
#include <utility>
#include <vector>
#include <iostream>
#include <memory.h>
#include <fstream>

namespace NeuralNetwork {
using Matrix = Eigen::MatrixXd;
using Vector = Eigen::VectorXd;
using Batch = std::vector<std::pair<Vector, Vector>>;

class DataLoader {
public:
    DataLoader(const std::string& path_to_images, const std::string& path_to_labels, size_t batch_size)
        : path_to_images(path_to_images), path_to_labels(path_to_labels), batch_size(batch_size) {
        file_images = std::ifstream(path_to_images, std::ios::binary);

        if (!file_images.is_open()) {
            throw std::runtime_error("Error opening file");
        }

        uint32_t magic_number;
        file_images.read(reinterpret_cast<char*>(&magic_number), 4);
        file_images.read(reinterpret_cast<char*>(&num_images), 4);
        file_images.read(reinterpret_cast<char*>(&num_rows_), 4);
        file_images.read(reinterpret_cast<char*>(&num_cols_), 4);

        magic_number = __builtin_bswap32(magic_number);
        num_images = __builtin_bswap32(num_images);
        num_rows_ = __builtin_bswap32(num_rows_);
        num_cols_ = __builtin_bswap32(num_cols_);
        if (magic_number != 0x00000803) {
            throw std::runtime_error("Invalid file format");
        }

        file_labels = std::ifstream(path_to_labels, std::ios::binary);
        if (!file_labels.is_open()) {
            std::cerr << "Ошибка при открытии файла " << path_to_labels << std::endl;
        }
        uint32_t magic_number_label = 0, num_items = 0;
        file_labels.read(reinterpret_cast<char*>(&magic_number_label), 4);
        file_labels.read(reinterpret_cast<char*>(&num_items), 4);
        magic_number_label = __builtin_bswap32(magic_number_label);
        if (magic_number_label != 2049) {
            std::cerr << "Неправильный формат меток файла " << path_to_labels << std::endl;
        }

        actual_index = 0;
    }
    Batch Next() {
        Batch batch(std::min(num_images - actual_index, batch_size));
        for (auto & i : batch) {
            i = std::make_pair(LoadImage(), ConvertInt(LoadLabel()));
            actual_index++;
        }
        return batch;
    }
    static Vector ConvertInt(int number) {
        Vector y = Eigen::Vector<double, 10>();
        y.setZero();
        y[number] = 1;
        return y;
    }
    static int ConvertVector(const Vector& y) {
        for (int i = 0; i < 10; ++i) {
            if (fabs(y[i] - 1) < 1e-6) {
                return i;
            }
        }
    }
    void Reset() {
        actual_index = 0;
        file_images.seekg(16, std::ios_base::beg);
        file_labels.seekg(8, std::ios::beg);
    }
private:
    Eigen::Vector<double, 784> LoadImage() {
        if (actual_index < 0 || actual_index >= num_images) {
            throw std::runtime_error("Invalid image index");
        }
        Eigen::Vector<double, 784> result;
        int cnt = 0;
        for (int i = 0; i < num_rows_ * num_cols_; i++) {
            unsigned char temp = 0;
            file_images.read((char *)&temp, sizeof(temp));
            cnt += temp;
            result(i) = (double)temp / 255.0;
        }
        return result;
    }

    uint8_t LoadLabel() {
        if (actual_index >= num_images) {
            std::cerr << "Индекс выходит за пределы диапазона в файле " << path_to_labels << std::endl;
            return 0;
        }
        uint8_t label = 0;
        file_labels.read(reinterpret_cast<char*>(&label), 1);
        return label;
    }

public:
    ~DataLoader() {
        file_images.close();
        file_labels.close();
    }
    int num_rows_;
    int num_cols_;
    std::ifstream file_images;
    std::ifstream file_labels;
    std::string path_to_images;
    std::string path_to_labels;
    size_t actual_index = 0;
    size_t num_images;
    size_t batch_size;
};

}
#endif //NEURALNETWORK_SRC_DATALOADER_H_
