import ast
import re
import numpy as np
from pandas import read_csv
from scipy.stats import ttest_ind
from test.experiments import statistics


def mean_acc_classes(base_file_name: str, col_name: str = 'classes'):
    data = read_csv(str(statistics.PATH / base_file_name) + '.csv', sep=';')
    classes = np.array([ast.literal_eval(
        re.sub(r' \[\[', r'[', re.sub(r', \[', ', ', re.sub(r', [0-9]*\]', '', re.sub('[0-9]*] ', '', row))))) for row
        in data[col_name]])
    return np.mean(classes, axis=0)


def generate_table(file_names: list[str], file_save: str, names: list[str] = None):
    """
    Generate latex table entries for comparison
    """
    file_names = [file_name + '.csv' for file_name in file_names]
    data = [read_csv(statistics.PATH / file_name, sep=';') for file_name in file_names]
    acc = 'acc'
    f1 = 'f1'
    result = ['test & accuracy & f1-measure & nothing & pair & two of a kind & three of a kind & straight & flush &'
              ' full house & four of a kind & straight flush & royal flush']
    for i, d in enumerate(data):
        classes = np.array([ast.literal_eval(
            re.sub(r' \[\[', r'[', re.sub(r', \[', ', ', re.sub(r', [0-9]*\]', '', re.sub('[0-9]*] ', '', row))))) for
            row in d['classes']])
        mean_classes_acc = np.mean(classes, axis=0)
        string_classes = ''.join((' & ' + str(round(mean_class_acc, 3)) for mean_class_acc in mean_classes_acc))
        string = names[i] + ' & ' + str(round(np.mean(d[acc]),3)) + ' & ' + str(round(np.mean(d[f1]),3)) + string_classes + '\\\\'
        result.append(string)
    with open(str(statistics.PATH / file_save) + '.csv', 'w') as f:
        for row in result:
            f.write("%s\n" % row)


def student_t_test(file_name1: str, file_name2: str, attribute: str = 'acc'):
    file_name1 = file_name1 + '.csv'
    data1 = read_csv(statistics.PATH / file_name1, sep=';')

    file_name2 = file_name2 + '.csv'
    data2 = read_csv(statistics.PATH / file_name2, sep=';')

    return ttest_ind(data1[attribute], data2[attribute])


def student_t_test_class(file_name1: str, file_name2: str, class_index: int = 0, col_name: str = 'classes'):
    """
    Compute student-t test on single classes for two different populations
    """
    file_name1 = file_name1 + '.csv'
    data1 = read_csv(statistics.PATH / file_name1, sep=';')
    classes1 = np.array([ast.literal_eval(
        re.sub(r' \[\[', r'[', re.sub(r', \[', ', ', re.sub(r', [0-9]*\]', '', re.sub('[0-9]*] ', '', row))))) for row
        in data1[col_name]])

    file_name2 = file_name2 + '.csv'
    data2 = read_csv(statistics.PATH / file_name2, sep=';')
    classes2 = np.array([ast.literal_eval(
        re.sub(r' \[\[', r'[', re.sub(r', \[', ', ', re.sub(r', [0-9]*\]', '', re.sub('[0-9]*] ', '', row))))) for row
        in data2[col_name]])

    return ttest_ind(classes1[:, class_index], classes2[:, class_index])


