module;

#include <concepts>

export module gsml.nn.loss.mse;

namespace gsml {
namespace loss {
// Concepts for static matrices
template<typename T>
concept IsMatrixStatic = requires(T t) {
    { t.array() } -> std::same_as<typename T::Array>;
    { t.array().square() } -> std::same_as<typename T::Array>;
    { t.array().sum() } -> std::convertible_to<float>;
    { t.size() } -> std::convertible_to<std::size_t>;
};

// Concepts for dynamic matrices
template<typename T>
concept IsMatrixDynamic = requires(T t) {
    { t.rows() } -> std::convertible_to<int>;
    { t.cols() } -> std::convertible_to<int>;
    { t.size() } -> std::convertible_to<size_t>;
};

export template<typename T, typename U = float>
requires IsMatrixStatic<T> or IsMatrixDynamic<T>
constexpr U mse(const T& prediction, const T& target) {
    const T error = prediction - target;
    return (error.array().square().sum() / error.size());
}

export template<typename T>
requires IsMatrixStatic<T> or IsMatrixDynamic<T>
constexpr T mse_gradient(const T& prediction, const T& target) {
    const T error = prediction - target;
    return 2.0 * error / prediction.size();
}
}
} // namespace gsml