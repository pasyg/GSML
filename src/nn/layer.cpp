module;

#include <random>

#include "eigen3/Eigen/Dense"

export module gsml.nn.layer;

namespace gsml {
export template<typename T = float, int Input = Eigen::Dynamic, int Output = Eigen::Dynamic>
struct Layer {
    using WeightMatrix = Eigen::Matrix<T, Input, Output>;
    using InputVector  = Eigen::Matrix<T, Input, 1>;
    using OutVector    = Eigen::Matrix<T, Output, 1>;
    
    virtual OutVector forward(const InputVector& input) = 0;
    virtual WeightMatrix backward(const WeightMatrix& gradient) = 0;

    const int rows() const noexcept {
        return static_cast<int>(this->weights.rows());
    }

    const int cols() const noexcept {
        return static_cast<int>(this->weights.cols());
    }

    virtual void initialize() {
        static std::random_device rd;
        static std::mt19937 gen(rd());
        static std::uniform_real_distribution<> dis(-1.0, 1.0);
        auto random_fill = [&]{return dis(gen);};

        if constexpr(Input == Eigen::Dynamic) {
            this->weights = WeightMatrix::NullaryExpr(this->rows(), this->cols(), random_fill);
            this->bias    = OutVector::NullaryExpr(this->rows(), random_fill);
        } else {
            this->weights = WeightMatrix::NullaryExpr(random_fill);
            this->bias    = OutVector::NullaryExpr(random_fill);
        }
    };

    WeightMatrix weights;
    OutVector bias;
};
}