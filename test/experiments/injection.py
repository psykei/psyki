import itertools
from keras import Model
from tensorflow.keras import Input
from tensorflow.keras.activations import softmax
from keras.optimizer_v2.adam import Adam
from psyki import Injector
from test import POKER_RULES, get_mlp, train_network, get_processed_dataset

EXPERIMENTS: int = 30
EPOCHS: list[int] = [100, ]
LAYERS: list[int] = [3, ]
NEURONS: list[int] = [128, ]
BATCH_SIZE: list[int] = [32, ]
INJECTION: bool = True
OPTIMIZER = Adam(learning_rate=0.001)

prefix = 'R2/'
train_x, train_y, test_x, test_y = get_processed_dataset('poker', validation=0.05)

for epochs, layers, neurons, batch_size in itertools.product(EPOCHS, LAYERS, NEURONS, BATCH_SIZE):
    for i in range(EXPERIMENTS):
        net_input = Input((10,), name='Input')
        network = get_mlp(net_input, output=10, layers=layers, neurons=neurons, activation_function='relu',
                          last_activation_function='softmax')
        if INJECTION:
            file = prefix + 'injection_L' + str(layers) + '_N' + str(neurons) + '_E' + str(epochs) + '_B' + \
                   str(batch_size) + '_I' + str(i + 1)
            injector = Injector(network, net_input, softmax)
            injector.inject(POKER_RULES)
            injector.predictor.compile(OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model = injector.predictor

        else:
            file = prefix + 'classic_L' + str(layers) + '_N' + str(neurons) + '_E' + str(epochs) + '_B' + \
                   str(batch_size) + '_I' + str(i + 1)
            model = Model(net_input, network)
            model.compile(OPTIMIZER, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Train the model with rules
        train_network(model, train_x, train_y, test_x, test_y, batch_size=batch_size, epochs=epochs, file=file)
