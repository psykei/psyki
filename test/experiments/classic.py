import itertools
from keras import Model
from tensorflow.keras import Input
from keras.optimizer_v2.adam import Adam
from test import get_mlp, train_network, get_processed_dataset, save_network

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
    optimizer = Adam(learning_rate=0.001)
    model = Model(input, network)
    model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # print('Neural Network With Knowledge Model Summary: ')
    # print(injector.predictor.summary())
    # plot_model(injector.predictor, to_file=str(img.PATH / 'model.png'))

    # Train the model with rules
    file = 'classic_L' + str(layers) + '_N' + str(neurons) + '_E' + str(epochs) + '_B' + str(batch_size)
    train_network(model, train_x, train_y, test_x, test_y, batch_size=batch_size, epochs=epochs, file=file)
    save_network(model, file)

"""
results = model.evaluate(test_x, test_y)
c_accuracy = class_accuracy(model, test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
print(c_accuracy)
"""
