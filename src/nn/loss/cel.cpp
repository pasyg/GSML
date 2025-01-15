module;

import gsml.concepts.matrix;

export module gsml.nn.loss.cel;

namespace gsml {
namespace loss {
export template <typename T, typename U = float>
requires CELStatic<T> or CELDynamic<T>
U cel(const T& prediction, const T& target) {
    // TODO: implement
    return 1.0;
}

export template <typename T>
requires CELStatic<T> or CELDynamic<T>
T cel_gradient(const T& prediction, const T& target) {
    // TODO: implement
    return prediction;
}

} // namespace loss
} // namespace gsml