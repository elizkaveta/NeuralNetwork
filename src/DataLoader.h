#ifndef NEURALNETWORK_SRC_DATALOADER_H_
#define NEURALNETWORK_SRC_DATALOADER_H_

#endif //NEURALNETWORK_SRC_DATALOADER_H_

#include "../eigen/Eigen/Dense"
#include "MNIST.h"
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
    DataLoader(const std::string &path_to_images, const std::string &path_to_labels, size_t batch_size);

    ~DataLoader();

private:
    friend class Model;
    void Next(Batch &batch);

    void Reset();

    Eigen::Vector<double, MNIST::IMAGE_SIZE> LoadImage();

    uint8_t LoadLabel();

    size_t batch_size;
    size_t size_of_picture;
    std::ifstream file_images;
    std::ifstream file_labels;
    size_t actual_index = 0;
    uint32_t num_images;
};

} // namespace NeuralNetwork
