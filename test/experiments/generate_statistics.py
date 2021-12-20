import glob
import os
import pandas as pd
from matplotlib import pyplot as plt
from test.experiments import statistics, img

files = glob.glob(str(statistics.PATH / '*.csv'))


def read_statistic(file: str) -> pd.DataFrame:
    return pd.read_csv(str(statistics.PATH / file), sep=';', index_col=0)


def save_plot(files: list[str]):
    for file in files:
        history = read_statistic(file)
        plt.plot(history['accuracy'])
        plt.plot(history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train acc', 'val acc'], loc='upper left')
        plot_name = str(img.PATH / os.path.basename(file)[:-4]) + '_accuracy.png'
        if not os.path.isfile(plot_name):
            plt.savefig(plot_name)
        plt.cla()

        plt.plot(history['loss'])
        plt.plot(history['val_loss'])
        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(['train loss', 'val loss'], loc='upper left')
        plot_name = str(img.PATH / os.path.basename(file)[:-4]) + '_loss.png'
        if not os.path.isfile(plot_name):
            plt.savefig(plot_name)
        plt.cla()


save_plot(files)
