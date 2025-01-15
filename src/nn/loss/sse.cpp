module;

import gsml.concepts.matrix;

export module gsml.nn.loss.sse;

namespace gsml {
namespace loss {
export template<typename T, typename U = float>
requires IsMatrixStatic<T> or IsMatrixDynamic<T>
constexpr U sse(const T& prediction, const T& target) {
    const T error = prediction - target;
    return error.array().square().sum();
}

export template<typename T>
requires IsMatrixStatic<T> or IsMatrixDynamic<T>
constexpr T sse_grad(const T& prediction, const T& target) {
    const T error = prediction - target;
    return 2.0 * error;
}
}
} // namespace gsml