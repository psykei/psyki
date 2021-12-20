import itertools
from tensorflow.keras import Input
from tensorflow.keras.activations import softmax
from keras.optimizer_v2.adam import Adam
from psyki import Injector
from test import POKER_RULES, get_mlp, train_network, get_processed_dataset, save_network_from_injector

EPOCHS: list[int] = [100, ]
LAYERS: list[int] = [3, ]
NEURONS: list[int] = [10, 50, 100, 200]
BATCH_SIZE: list[int] = [50, 100]


train_x, train_y, test_x, test_y = get_processed_dataset('poker')

# Build the model
for epochs, layers, neurons, batch_size in itertools.product(EPOCHS, LAYERS, NEURONS, BATCH_SIZE):
    input = Input((10,), name='Input')
    network = get_mlp(input, output=10, layers=layers, neurons=neurons, activation_function='relu',
                      last_activation_function='softmax')
    injector = Injector(network, input, softmax)
    injector.inject(POKER_RULES)
    optimizer = Adam(learning_rate=0.001)
    injector.predictor.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # print('Neural Network With Knowledge Model Summary: ')
    # print(injector.predictor.summary())
    # plot_model(injector.predictor, to_file=str(img.PATH / 'model.png'))

    # Train the model with rules
    file = 'model_L' + str(layers) + '_N' + str(neurons) + '_E' + str(epochs) + '_B' + str(batch_size)
    train_network(injector.predictor, train_x, train_y, test_x, test_y, batch_size=batch_size, epochs=epochs, file=file)
    save_network_from_injector(injector, file)
