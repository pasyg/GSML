module;

#include "eigen3/Eigen/Dense"

import gsml.nn.layer;

export module gsml.nn.conv;

// TODO: nothing implemented.

namespace gsml {
export template<typename T = float, 
                int Input = Eigen::Dynamic, 
                int Output = Eigen::Dynamic>
struct Conv : public Layer<T, Input, Output> {
    using WeightMatrix = Eigen::Matrix<T, Input, Output>;
    using InputVector  = Eigen::Matrix<T, Input, 1>;
    using OutVector    = Eigen::Matrix<T, Output, 1>;

    constexpr Conv() {}

    Conv(int input, int output) {
        this->weights = WeightMatrix();
    }

    OutVector forward(const InputVector& input) override {
        // TODO not implemented
        return this->weights;
    }

    WeightMatrix backward(const WeightMatrix& gradient) override {
        // TODO not implemented
        return this->weights;
    }
};
} // namespace gsml