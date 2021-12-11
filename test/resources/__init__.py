import csv
from pathlib import Path

PATH = Path(__file__).parents[0]


def get_rules(name: str) -> dict[str, str]:
    result = {}
    with open(str(PATH / name) + '.csv', mode="r") as file:
        reader = csv.reader(file, delimiter=';')
        for item in reader:
            result[item[0]] = item[1]
    return result
