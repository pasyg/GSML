module;

import gsml.concepts.matrix;

export module gsml.nn.loss.mse;


namespace gsml {
namespace loss {
export template<typename T, typename U = float>
requires IsMatrixStatic<T> or IsMatrixDynamic<T>
constexpr U mse(const T& prediction, const T& target) {
    const T error = prediction - target;
    return (error.array().square().sum() / error.size());
}

export template<typename T>
requires IsMatrixStatic<T> or IsMatrixDynamic<T>
constexpr T mse_grad(const T& prediction, const T& target) {
    const T error = prediction - target;
    return 2.0 * error / prediction.size();
}
}
} // namespace gsml