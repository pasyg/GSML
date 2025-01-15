module;

#include <iostream>
#include "eigen3/Eigen/Dense"

import gsml.nn.layer;

export module gsml.nn.activation.relu;

namespace gsml {
namespace activation {
export template<typename T>
auto relu(const T& x) {
    return x.unaryExpr([](const float val){
        return val > 0.0 ? val : 0.0;
    });
}

export template<typename T>
auto reluDerivative(const T& x) {
    return x.unaryExpr([](const float val){
        return val > 0.0 ? 1.0 : 0.0;
    });
}
}
} // namespace gsml