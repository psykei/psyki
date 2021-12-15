import csv
from pathlib import Path
from typing import Any
import numpy as np
from test.resources import rules, data

PATH = Path(__file__).parents[0]


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
