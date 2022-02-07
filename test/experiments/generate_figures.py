import ast
import itertools
import os
import re
import numpy as np
import pandas as pd
from colour import Color
from matplotlib import pyplot as plt
from matplotlib.pyplot import axes
from pandas import read_csv
from test.experiments import statistics, img


def read_history(file: str) -> pd.DataFrame:
    return pd.read_csv(str(statistics.PATH / file), sep=';', index_col=0)


def save_plot(files: list[str]):
    """
    Generate plot for network accuracy and loss during training.
    """
    for file in files:
        history = read_history(file)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train acc', 'val acc'], loc='lower right')
        plot_name = str(img.PATH / os.path.basename(file)[:-4]) + '_accuracy.png'
        plt.grid()
        plt.savefig(plot_name)
        plt.cla()

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'], loc='upper right')
        plot_name = str(img.PATH / os.path.basename(file)[:-4]) + '_loss.png'
        plt.grid()
        plt.savefig(plot_name)
        plt.cla()


def history_plot_comparison(files: list[str], title: str, colors, names: list[str] = None, min_exp: int = 0, max_exp: int = 30):
    """
    Generate plot for network accuracy and loss during training for two different set of experiments.
    """

    global_history = []
    single_history = None
    for file in files:
        for i in range(min_exp, max_exp):
            file2 = file + '_' + str(i) + '.csv'
            if file[-13:-6] == 'classic':
                file2 = file + str(i+1) + '.csv'
            single_history = read_history(file2) if single_history is None else single_history + read_history(file2)
        global_history.append(single_history / (max_exp - min_exp))
        single_history = None

    for i, _ in enumerate(files):
        plt.plot(global_history[i]['accuracy'], color=colors[i])
    plt.title('model accuracy')
    plt.ylabel('val_acc')
    plt.xlabel('epoch')
    # plt.legend(['train knowledge acc', 'train classic acc', 'val knowledge acc', 'val classic acc'], loc='lower right')
    plot_name = str(img.PATH / title) + '.pdf'
    plt.grid()
    plt.savefig(plot_name, format='pdf')
    plt.cla()


def box_plot(data, positions, ax, edge_color, fill_color=None):
    """
    Generate a boxplot with colors
    """
    bp = ax.boxplot(data, positions=positions, widths=0.8, patch_artist=True)

    for element in ['boxes', 'whiskers', 'fliers', 'means', 'medians', 'caps']:
        plt.setp(bp[element], color=edge_color)

    if fill_color is not None:
        for patch in bp['boxes']:
            patch.set(facecolor=fill_color)

    return bp


def classes_distribution(files: list[str], names: list[str], col_name: str = 'classes', class_indices: list[int] = None, fig_name: str = 'default', title: str = 'default', edge_colors: list[str] = None, fill_colors: list[str] = None):
    """
    Generate a plot with the accuracy class distribution of networks
    """
    file_names = [file_name + '.csv' for file_name in files]
    class_names = ['nothing', 'pair', 'two', 'three', 'straight', 'flush', 'full', 'four', 'straight f.', 'royal f.']
    class_names = [class_names[i] for i in class_indices]
    data = [read_csv(statistics.PATH / file_name, sep=';') for file_name in file_names]
    plt.figure()
    ax = axes()
    for j, k in enumerate(files):
        classes = np.array([ast.literal_eval(
            re.sub(r' \[\[', r'[', re.sub(r', \[', ', ', re.sub(r', [0-9]*\]', '', re.sub('[0-9]*] ', '', row)))))
            for row in data[j][col_name]])
        classes = classes[:, class_indices]
        bp =  box_plot(classes, positions=[len(file_names)*c+j+1 for c in range(len(class_names))], ax=ax, edge_color=edge_colors[j], fill_color=fill_colors[j])

    plt.title(title)
    plt.ylabel('accuracy')
    class_names = [file_name + ' (' + class_name + ')' for class_name, file_name in itertools.product(names, class_names)]
    ax.set_xticklabels(class_names, rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(str(img.PATH / fig_name) + '.pdf', format='pdf')
    plt.cla()


def metric_distribution(files: list[str], names: list[str], col_name='acc', classes=None, fig_name: str = 'default', title: str = 'default', edge_colors: list[str] = None, fill_colors: list[str] = None):
    """
    Generate a plot with the specified metric of networks
    """
    file_names = [file_name + '.csv' for file_name in files]
    data = [read_csv(statistics.PATH / file_name, sep=';') for file_name in file_names]
    plt.figure()
    ax = axes()
    for j, k in enumerate(files):
        scores = data[j][col_name]
        if classes is not None:
            scores = np.array([ast.literal_eval(
                re.sub(r' \[\[', r'[', re.sub(r', \[', ', ', re.sub(r', [0-9]*\]', '', re.sub('[0-9]*] ', '', row)))))
                for row in data[j]['classes']])[:,classes]
        bp = box_plot(scores, positions=[j+1], ax=ax, edge_color=edge_colors[j], fill_color=fill_colors[j])
    plt.title(title)
    plt.ylabel('Metric value')
    ax.set_xticklabels(names, rotation=90)
    plt.tight_layout()
    plt.grid()
    plt.savefig(str(img.PATH / fig_name) + '.pdf', format='pdf')
    plt.cla()


# save_plot([str(statistics.PATH / 'structuring4.csv')])

"""
experiments_file_names = ['test_results_classic', 'test_result_structuring7', 'test_result_structuring8', 'test_result_structuring9']
short_names = ['classic', 'Struct. 7 r.', 'Struct. 8 r.', 'Struct. 9 r.']
title = 'class accuracy distributions'
colors1 = ['red', 'blue', 'darkgreen', 'darkorange']
colors2 = ['salmon', 'cyan', 'lightgreen', 'bisque']

classes_distribution(experiments_file_names, short_names, 'classes', [0,1,2,3,4], 'classes-dist-struct-3-1',
                     title, colors1, colors2)
classes_distribution(experiments_file_names, short_names, 'classes', [5,6,7,8,9], 'classes-dist-struct-3-2',
                     title, colors1, colors2)
"""


def accuracy_boxplot_comparison(title, classes, file):
    experiments_file_names = ['test_results_classic'] + ['test_result_structuring' + str(i) for i in range(1, 11)][::-1]
    short_names = ['classic'] + ['R' + str(i) for i in range(1, 11)]
    colors1 = ['red'] + 10*['blue']
    colors2 = [color.hex for color in list(Color("salmon").range_to(Color("royalblue"),11))]

    metric_distribution(experiments_file_names, short_names, 'acc', classes, file, title, colors1, colors2)


names = ['nothing', 'pair', 'two of a kind', 'three of a kind', 'straight', 'flush', 'full house', 'four of a kind',
         'straight flush', 'royal flush']
titles = ['Accuracy distributions for class ' + name for name in names]
files = ['acc-dist-' + name for name in names]
for i in range(10):
    accuracy_boxplot_comparison(titles[i], i, files[i])

'''colors = [color.hex for color in list(Color("red").range_to(Color("blue"), 11))]
experiments_file_names = ['classic/model'] + ['structuring1' + str(i) + '/model' for i in range(1, 11)]
# short_names = ['classic'] + ['R' + str(i) for i in range(1, 11)]
title = 'Accuracy distributions'
history_plot_comparison(experiments_file_names, title, colors)'''