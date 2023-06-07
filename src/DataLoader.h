#ifndef NEURALNETWORK_SRC_DATALOADER_H_
#define NEURALNETWORK_SRC_DATALOADER_H_

#endif //NEURALNETWORK_SRC_DATALOADER_H_

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
    DataLoader(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size)
        : batch_size(batch_size) {
        file_images = std::ifstream(path_to_images, std::ios::binary);

        if (!file_images.is_open()) {
            throw std::runtime_error("Ошибка при открытии файла");
        }

        uint32_t magic_number_image, num_rows, num_cols;
        file_images.read(reinterpret_cast<char *>(&magic_number_image), POINTER);
        file_images.read(reinterpret_cast<char *>(&num_images), POINTER);
        file_images.read(reinterpret_cast<char *>(&num_rows), POINTER);
        file_images.read(reinterpret_cast<char *>(&num_cols), POINTER);

        magic_number_image = __builtin_bswap32(magic_number_image);
        num_images = __builtin_bswap32(num_images);
        num_rows = __builtin_bswap32(num_rows);
        num_cols = __builtin_bswap32(num_cols);

        size_of_picture = num_cols * num_rows;

        if (size_of_picture != IMAGE_SIZE) {
            std::cerr << "Неправильный формат файла " << path_to_labels << std::endl;
        }

        if (magic_number_image != MAGIC_NUMBER_IMAGE) {
            throw std::runtime_error("Неправильный формат файла");
        }

        file_labels = std::ifstream(path_to_labels, std::ios::binary);
        if (!file_labels.is_open()) {
            std::cerr << "Ошибка при открытии файла " << path_to_labels << std::endl;
        }

        uint32_t magic_number_label = 0, num_items = 0;
        file_labels.read(reinterpret_cast<char *>(&magic_number_label), POINTER);
        file_labels.read(reinterpret_cast<char *>(&num_items), POINTER);
        magic_number_label = __builtin_bswap32(magic_number_label);

        if (magic_number_label != MAGIC_NUMBER_LABEL) {
            std::cerr << "Неправильный формат меток файла " << path_to_labels << std::endl;
        }
        actual_index = 0;
    }

    ~DataLoader() {
        file_images.close();
        file_labels.close();
    }

    static const size_t IMAGE_SIZE = 784;
    static const int COUNT_OF_DIGITS = 10;

private:
    friend class Model;
    void Next(Batch &batch) {
        batch.resize(std::min(num_images - actual_index, batch_size));
        for (int i = 0; i < batch.size(); ++i) {
            batch[i] = std::make_pair(LoadImage(), ConvertInt(LoadLabel()));
            actual_index++;
        }
    }

    static Vector ConvertInt(int number) {
        Vector y = Eigen::Vector<double, COUNT_OF_DIGITS>();
        y.setZero();
        y[number] = 1;
        return y;
    }

    static int ConvertVector(const Vector &y) {
        int index = 0;
        for (int i = 0; i < COUNT_OF_DIGITS; ++i) {
            if (y[i] > y[index]) {
                index = i;
            }
        }
        return index;
    }

    void Reset() {
        actual_index = 0;
        file_images.seekg(4 * POINTER, std::ios_base::beg); // первые 4 числа файла заняты параметрами
        file_labels.seekg(2 * POINTER, std::ios::beg); // первые 2 числа файла заняты параметрами
    }

    Eigen::Vector<double, IMAGE_SIZE> LoadImage() {
        if (actual_index < 0 || actual_index >= num_images) {
            throw std::runtime_error("Индекс выходит за пределы диапазона");
        }

        Vector result(784);

        for (int i = 0; i < size_of_picture; i++) {
            unsigned char temp = 0;
            file_images.read((char *) &temp, sizeof(temp));
            // result(i) = std::max(std::min(((double)temp  - PIXEL_MAX / 2) * 0.96 + PIXEL_MAX / 2, PIXEL_MAX), 0.0) / PIXEL_MAX;
            result(i) = temp / PIXEL_MAX;
        }
        return result;
    }

    uint8_t LoadLabel() {
        if (actual_index >= num_images) {
            std::cerr << "Индекс выходит за пределы диапазона" << std::endl;
            return 0;
        }
        uint8_t label = 0;
        file_labels.read(reinterpret_cast<char *>(&label), 1);
        return label;
    }

    const std::streamsize POINTER = 4;
    const double PIXEL_MAX = 255.0;
    const int MAGIC_NUMBER_IMAGE = 0x00000803;
    const int MAGIC_NUMBER_LABEL = 2049;

    size_t batch_size;
    size_t size_of_picture;
    std::ifstream file_images;
    std::ifstream file_labels;
    size_t actual_index = 0;
    uint32_t num_images;
};

} // namespace NeuralNetwork
