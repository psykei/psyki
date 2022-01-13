import ast
import itertools
import os
import re
import numpy as np
import pandas as pd
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
        plt.legend(['train acc', 'val acc'], loc='bottom right')
        plot_name = str(img.PATH / os.path.basename(file)[:-4]) + '_accuracy.png'
        if not os.path.isfile(plot_name):
            plt.savefig(plot_name)
        plt.cla()

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'], loc='upper right')
        plot_name = str(img.PATH / os.path.basename(file)[:-4]) + '_loss.png'
        if not os.path.isfile(plot_name):
            plt.savefig(plot_name)
        plt.cla()


def save_plot_comparison(base_file_name1: str, base_file_name2: str, min_exp: int = 0, max_exp: int = 30):
    """
    Generate plot for network accuracy and loss during training for two different set of experiments.
    """

    knowledge_history = None
    classic_history = None

    for i in range(min_exp, max_exp):
        file_exp1 = base_file_name1 + '_I' + str(i + 1) + '.csv'
        file_exp2 = base_file_name2 + '_I' + str(i + 1) + '.csv'
        knowledge_history = read_history(file_exp1) if knowledge_history is None else knowledge_history + read_history(file_exp1)
        classic_history = read_history(file_exp2) if classic_history is None else classic_history + read_history(file_exp2)

    knowledge_history = knowledge_history / (max_exp - min_exp)
    classic_history = classic_history / (max_exp - min_exp)
    plt.plot(knowledge_history['accuracy'])
    plt.plot(classic_history['accuracy'])
    plt.plot(knowledge_history['val_accuracy'])
    plt.plot(classic_history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train knowledge acc', 'train classic acc', 'val knowledge acc', 'val classic acc'],
               loc='lower right')
    plot_name = str(img.PATH / os.path.basename(base_file_name1.replace('model', 'comparison', 1))[:-4]) + '_accuracy.png'
    plt.grid()
    plt.savefig(plot_name)
    plt.cla()

    plt.plot(knowledge_history['loss'])
    plt.plot(classic_history['loss'])
    plt.plot(knowledge_history['val_loss'])
    plt.plot(classic_history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train knowledge loss', 'train classic loss', 'val knowledge loss', 'val classic loss'],
               loc='upper right')
    plot_name = str(img.PATH / os.path.basename(base_file_name1.replace('model', 'comparison', 1))[:-4]) + '_loss.png'
    plt.grid()
    if not os.path.isfile(plot_name):
        plt.savefig(plot_name)
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


def classes_distribution(files: list[str], names: list[str], col_name: str = 'knowledge_classes', class_indices: list[int] = None, fig_name: str = 'default', title: str = 'default', edge_colors: list[str] = None, fill_colors: list[str] = None):
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
    plt.savefig(str(img.PATH / fig_name) + '.svg', format='svg')
    plt.cla()


experiments_file_names = ['test_results_classic', 'test_results_R1', 'test_results_R2']
short_names = ['classic', 'R1', 'R2']
title = 'class accuracy distributions'
colors1 = ['red','blue', 'darkgreen', 'darkorange']
colors2 = ['salmon', 'cyan', 'lightgreen', 'bisque']
classes_distribution(experiments_file_names, short_names, 'knowledge_classes', [0,1,2,3,4], 'classes_distribution1',
                     title, colors1, colors2)
classes_distribution(experiments_file_names, short_names, 'knowledge_classes', [5,6,7,8,9], 'classes_distribution2',
                     title, colors1, colors2)