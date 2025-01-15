#include <cstdlib>
#include <iostream>

#include "Eigen/Eigen"

import gsml.nn.dense;
import gsml.nn.activation.relu;
import gsml.nn.loss.mse;

#include <random>

// Will get better, just for testing purposes right now
Eigen::MatrixXf rand_matrix(const int rows, const int cols) {
    std::random_device rd;
    std::mt19937 generator(rd());
    std::normal_distribution<float> distribution(0);
    auto normal = [&] () {return distribution(generator);};

    Eigen::MatrixXf m = Eigen::MatrixXf::NullaryExpr(rows, cols, normal);
    return m;
}

int main() {
    std::cout << "Eigen Version: " << EIGEN_WORLD_VERSION << "." << EIGEN_MAJOR_VERSION << "." << EIGEN_MINOR_VERSION << std::endl;

    auto layer = gsml::Dense<float, 2, 3>();
    layer.initialize();

    auto input = rand_matrix(2, 1);
    std::cout << "Input:\n" << input << std::endl;

    auto activatad = gsml::activation::reluDerivative(input);
    std::cout << "Input after ReLuDerivative:\n" << activatad << std::endl;

    auto out = layer.forward(input); 
    std::cout << "Output:\n" << out << std::endl;
    
    auto activated = gsml::activation::relu(out);
    std::cout << "Output after ReLu:\n" << activated << std::endl;

    auto activetad = gsml::activation::reluDerivative(out);
    std::cout << "Output after ReLuDerivative:\n" << activetad << std::endl;

    auto prediction = rand_matrix(5, 3);
    auto target     = rand_matrix(5, 3);

    auto loss = gsml::loss::mse(prediction, target);
    auto grad = gsml::loss::mse_gradient(prediction, target);

    std::cout << "Loss: " << loss << std::endl;
    std::cout << "Gradient:\n" << grad << std::endl;

    return 0;
};