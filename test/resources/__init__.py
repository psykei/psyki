import csv
from pathlib import Path
from typing import Any
import numpy as np
from tensorflow import constant, float32
from test.resources import rules, data

PATH = Path(__file__).parents[0]

POKER_INPUT_MAPPING = {
        'S1': 0,
        'R1': 1,
        'S2': 2,
        'R2': 3,
        'S3': 4,
        'R3': 5,
        'S4': 6,
        'R4': 7,
        'S5': 8,
        'R5': 9
    }

POKER_OUTPUT_MAPPING = {
        'nothing':          constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float32),
        'pair':             constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=float32),
        'twoPairs':         constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=float32),
        'tris':             constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=float32),
        'straight':         constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float32),
        'flush':            constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=float32),
        'full':             constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=float32),
        'poker':            constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=float32),
        'straightFlush':    constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=float32),
        'royalFlush':       constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=float32)
    }

def get_rules(name: str) -> dict[str, str]:
    result = {}
    with open(str(rules.PATH / name) + '.csv', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result[item[0]] = item[1]
    return result


def get_ordered_rules(name: str) -> list[str]:
    result = []
    with open(str(rules.PATH / name) + '.csv', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result.append(item[1])
    return result


def get_dataset(name: str) -> Any:
    return np.genfromtxt(str(data.PATH / name) + '.csv', delimiter=',', dtype=float)
