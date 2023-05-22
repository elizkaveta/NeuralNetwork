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

    }

    std::pair<Vector, Vector> KeyValue(size_t index) {
        return std::make_pair(FlattenImage(LoadImage(index)), ConvertInt(LoadLabel(index)));
    }
    Batch Next() {
        Batch batch(std::min(num_images - actual_index, batch_size));
        for (size_t i = 0; i < batch.size(); ++i) {
            batch[i] = KeyValue(actual_index + i);
        }
        actual_index += batch.size();
        return batch;
    }
    static Vector ConvertInt(int number) {
        Vector y = Vector::Random(10);
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
private:
    Eigen::VectorXd FlattenImage(const Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>& image) {
        Eigen::Map<const Eigen::VectorXd> f_image(image.data(), image.size());
        return f_image;
    }

    Eigen::Matrix<double, 28, 28> LoadImage(int index) {
        std::ifstream file(path_to_images, std::ios::binary);

        if (!file.is_open()) {
            throw std::runtime_error("Error opening file");
        }

        uint32_t magic_number, num_images, num_rows, num_columns;
        file.read(reinterpret_cast<char*>(&magic_number), sizeof(magic_number));
        file.read(reinterpret_cast<char*>(&num_images), sizeof(num_images));
        file.read(reinterpret_cast<char*>(&num_rows), sizeof(num_rows));
        file.read(reinterpret_cast<char*>(&num_columns), sizeof(num_columns));

        magic_number = __builtin_bswap32(magic_number);
        num_images = __builtin_bswap32(num_images);

        if (magic_number != 0x00000803) {
            throw std::runtime_error("Invalid file format");
        }

        if (index < 0 || index >= num_images) {
            throw std::runtime_error("Invalid image index");
        }

        int image_offset = index * num_rows * num_columns;
        file.seekg(image_offset, std::ios_base::cur);
        Eigen::Matrix<double, 28, 28> img;
        for (int i = 0; i < 28; ++i) {
            for (int j = 0; j < 28; ++j) {
                uint8_t pixel;
                file.read(reinterpret_cast<char*>(&pixel), sizeof(pixel));
                img(i, j) = pixel / 255.0;
            }
        }

        file.close();
        return img;
    }

    uint8_t LoadLabel(int index) {
        std::ifstream file(path_to_labels, std::ios::binary);
        if (!file.is_open()) {
            std::cerr << "Ошибка при открытии файла " << path_to_labels << std::endl;
            return 0;
        }
        uint32_t magic_number = 0, num_items = 0;
        file.read(reinterpret_cast<char*>(&magic_number), 4);
        file.read(reinterpret_cast<char*>(&num_items), 4);
        magic_number = __builtin_bswap32(magic_number);
        num_items = __builtin_bswap32(num_items);
        num_images = static_cast<int>(num_items);
        if (magic_number != 2049) {
            std::cerr << "Неправильный формат меток файла " << path_to_labels << std::endl;
            return 0;
        }
        if (index >= num_images) {
            std::cerr << "Индекс выходит за пределы диапазона в файле " << path_to_labels << std::endl;
            return 0;
        }
        file.seekg(8 + index);
        uint8_t label = 0;
        file.read(reinterpret_cast<char*>(&label), 1);
        file.close();
        return label;
    }

public:
    std::string path_to_images;
    std::string path_to_labels;
    size_t actual_index = 0;
    size_t num_images;
    size_t batch_size;
};

}
#endif //NEURALNETWORK_SRC_DATALOADER_H_
