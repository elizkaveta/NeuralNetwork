#include "DataLoader.h"

namespace NeuralNetwork {

    DataLoader::DataLoader(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size)
        : batch_size(batch_size) {
        file_images = std::ifstream(path_to_images, std::ios::binary);

        if (!file_images.is_open()) {
            throw std::runtime_error("Ошибка при открытии файла");
        }

        uint32_t magic_number_image, num_rows, num_cols;
        file_images.read(reinterpret_cast<char *>(&magic_number_image), MNIST::POINTER);
        file_images.read(reinterpret_cast<char *>(&num_images), MNIST::POINTER);
        file_images.read(reinterpret_cast<char *>(&num_rows), MNIST::POINTER);
        file_images.read(reinterpret_cast<char *>(&num_cols), MNIST::POINTER);

        magic_number_image = __builtin_bswap32(magic_number_image);
        num_images = __builtin_bswap32(num_images);
        num_rows = __builtin_bswap32(num_rows);
        num_cols = __builtin_bswap32(num_cols);

        size_of_picture = num_cols * num_rows;

        if (size_of_picture != MNIST::IMAGE_SIZE) {
            throw std::runtime_error("Неправильный формат файла");
        }

        if (magic_number_image != MNIST::MAGIC_NUMBER_IMAGE) {
            throw std::runtime_error("Неправильный формат файла");
        }

        file_labels = std::ifstream(path_to_labels, std::ios::binary);
        if (!file_labels.is_open()) {
            throw std::runtime_error("Ошибка при открытии файла");
        }

        uint32_t magic_number_label = 0, num_items = 0;
        file_labels.read(reinterpret_cast<char *>(&magic_number_label), MNIST::POINTER);
        file_labels.read(reinterpret_cast<char *>(&num_items), MNIST::POINTER);
        magic_number_label = __builtin_bswap32(magic_number_label);

        if (magic_number_label != MNIST::MAGIC_NUMBER_LABEL) {
            throw std::runtime_error("Неправильный формат меток файла");
        }
        actual_index = 0;
    }

    DataLoader::~DataLoader() {
        file_images.close();
        file_labels.close();
    }

    void DataLoader::Next(Batch &batch) {
        batch.resize(std::min(num_images - actual_index, batch_size));
        for (auto & i : batch) {
            i = std::make_pair(LoadImage(), MNIST::ConvertInt(LoadLabel()));
            actual_index++;
        }
    }


    void DataLoader::Reset() {
        actual_index = 0;
        file_images.seekg(4 * MNIST::POINTER, std::ios_base::beg); // первые 4 числа файла заняты параметрами
        file_labels.seekg(2 * MNIST::POINTER, std::ios::beg); // первые 2 числа файла заняты параметрами
    }

    Eigen::Vector<double, MNIST::IMAGE_SIZE> DataLoader::LoadImage() {
        if (actual_index < 0 || actual_index >= num_images) {
            throw std::runtime_error("Индекс выходит за пределы диапазона");
        }

        Vector result(784);

        for (int32_t i = 0; i < size_of_picture; i++) {
            unsigned char temp = 0;
            file_images.read((char *) &temp, sizeof(temp));
            // result(i) = std::max(std::min(((double)temp  - PIXEL_MAX / 2) * 0.96 + PIXEL_MAX / 2, PIXEL_MAX), 0.0) / PIXEL_MAX;
            result(i) = temp / MNIST::PIXEL_MAX;
        }
        return result;
    }

    uint8_t DataLoader::LoadLabel() {
        if (actual_index >= num_images) {
            throw std::runtime_error("Индекс выходит за пределы диапазона");
        }
        uint8_t label = 0;
        file_labels.read(reinterpret_cast<char *>(&label), 1);
        return label;
    };

} // namespace NeuralNetwork
