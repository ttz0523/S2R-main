import random
from .sc_ng import SCNG


def get_random_float(float_range: [float]):
    return random.random() * (float_range[1] - float_range[0]) + float_range[0]


def get_random_int(int_range: [int]):
    return random.randint(int_range[0], int_range[1])


