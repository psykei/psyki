from tensorflow import Tensor, minimum, maximum


def eta(x: Tensor) -> Tensor:
    return minimum(1., maximum(0., x))
