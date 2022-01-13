import os
from keras import Input
from keras.activations import softmax
from keras.models import load_model, Model
from keras.optimizer_v2.adam import Adam
from psyki import Injector
from test import get_processed_dataset, class_accuracy, f1, get_mlp
from test.experiments import models, statistics


def save_model_without_lambda(base_file_name: str, min_exp: int = 0, max_exp: int = 30):
    """
    Remove lambda layer from saved network
    """
    optimizer = Adam(learning_rate=0.001)
    for i in range(min_exp, max_exp):
        file_exp = base_file_name + '_I' + str(i + 1) + '.h5'
        file_exp = str(models.PATH / file_exp)
        input = Input((10,), name='Input')
        network = get_mlp(input, output=10, layers=3, neurons=128, activation_function='relu',
                          last_activation_function='softmax')
        injector = Injector(network, input, softmax)
        model = injector.load(file_exp)
        model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        if len(model.layers) >= 6:
            model = Model(inputs=model.net_input, outputs=model.layers[-3].output)
            model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.save(file_exp)


def compute_performance(base_file_name: str, file_save: str = 'default', min_exp: int = 0, max_exp: int = 30, append: bool = False):
    info = [] if append else ["model;acc;f1;weighted_f1;classes"]
    optimizer = Adam(learning_rate=0.001)
    for i in range(min_exp, max_exp):
        file_exp = base_file_name + '_I' + str(i + 1) + '.h5'
        file_exp = str(models.PATH / file_exp)
        print(file_exp)
        model = load_model(file_exp)
        model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        classes_accuracy = class_accuracy(model, test_x, test_y)
        macro_f1 = f1(model, test_x, test_y)
        weighted_f1 = f1(model, test_x, test_y, 'weighted')
        info.append(
            os.path.basename(file_exp) + '; ' +
            str(model.evaluate(test_x, test_y)[1]) + '; ' +
            str(macro_f1) + '; ' +
            str(weighted_f1) + '; ' +
            str(classes_accuracy)
        )
    mode = 'a' if append else 'w'
    with open(str(statistics.PATH / file_save) + '.csv', mode) as f:
        for row in info:
            f.write("%s\n" % row)


_, _, test_x, test_y = get_processed_dataset('poker')
file = 'R2/model_L3_N128_E100_B32'

save_model_without_lambda(file, 0, 30)
compute_performance(file, 'test_results_R2', 0, 30, False)
