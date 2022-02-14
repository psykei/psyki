from tensorflow import Tensor, maximum, minimum, abs


def abstract_method_exception():
    raise Exception('Try to call an abstract method')


def parse_exception(string: str):
    raise Exception('Parser cannot parse the provided string: ' + string)


def eta(x: Tensor) -> Tensor:
    """
    :param x: a tensorflow tensor
    :return: the input tensor with all its elements scaled in range [0, 1]
    """
    return maximum(0., minimum(1., x))


def eta_abs(x: Tensor) -> Tensor:
    return eta(abs(x))


def eta_abs_one(x: Tensor) -> Tensor:
    return eta(abs(x - 1.))


def eta_one_abs(x: Tensor) -> Tensor:
    return eta(1. - abs(x))
