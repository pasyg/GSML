module;

#include <concepts>

export module gsml.concepts.matrix;


namespace gsml {
// Concept for static matrices
export template<typename T>
concept IsMatrixStatic = requires(T t) {
    { t.array() } -> std::same_as<typename T::Array>;
    { t.array().square() } -> std::same_as<typename T::Array>;
    { t.array().sum() } -> std::convertible_to<float>;
};

// Concept for dynamic matrices
export template<typename T>
concept IsMatrixDynamic = requires(T t) {
    { t.rows() } -> std::convertible_to<int>;
    { t.cols() } -> std::convertible_to<int>;
};

// Concept for cross entropy loss for static matrices
export template <typename T>
concept CELStatic = requires(T t) {
  { t.array() } -> std::same_as<typename T::Array>;
  { t.array().log() } -> std::same_as<typename T::Array>; 
  { t.size() } -> std::convertible_to<std::size_t>;
  { t.array() * t.array() } -> std::same_as<typename T::Array>;
  { t.array() + t.array() } -> std::same_as<typename T::Array>;
};

// Concept for cross entropy loss for dynamic matrices
export template <typename T>
concept CELDynamic = requires(T t) {
    { t.rows() } -> std::convertible_to<int>;
    { t.cols() } -> std::convertible_to<int>;
    { t.rows() } -> std::convertible_to<size_t>;
};
}