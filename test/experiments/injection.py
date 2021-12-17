from keras import Model
from keras.callbacks import CSVLogger
from keras.utils.vis_utils import plot_model
from tensorflow.keras import Input
from tensorflow.keras.activations import softmax
from sklearn.preprocessing import OneHotEncoder
from keras.optimizer_v2.adam import Adam
from psyki import Injector
from test import POKER_RULES, class_accuracy, get_mlp, train_network
from test.resources import get_dataset

EPOCHS: int = 80
NEURONS: int = 100
BATCH_SIZE: int = 50

poker_training = get_dataset('poker-training')
poker_testing = get_dataset('poker-testing')

train_x = poker_training[:, :-1]
train_y = poker_training[:, -1]
test_x = poker_training[:, :-1]
test_y = poker_training[:, -1]

# One Hot encode the class labels
encoder = OneHotEncoder(sparse=False)
encoder.fit_transform([train_y])
encoder.fit_transform([test_y])

# Build the model

input = Input((10,))
network = get_mlp(input, output=10, layers=3, neurons=NEURONS, activation_function='relu', last_activation_function='softmax')
injector = Injector(network, input, softmax)
injector.inject(POKER_RULES)
optimizer = Adam(learning_rate=0.001)
injector.predictor.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Neural Network With Knowledge Model Summary: ')
print(injector.predictor.summary())
plot_model(injector.predictor)

# Train the model with rules
train_network(injector.predictor, train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Disable rules
injector.knowledge = False
results = injector.predictor.evaluate(test_x, test_y)
c_accuracy = class_accuracy(injector.predictor, test_x, test_y)

print('Final test set loss after removing rules: {:4f}'.format(results[0]))
print('Final test set accuracy after removing rules: {:4f}'.format(results[1]))
print(c_accuracy)

injector.predictor.save('knowledgeNN80Epochs')

input = Input((10,))
network = get_mlp(input, output=10, layers=3, neurons=NEURONS, activation_function='relu', last_activation_function='softmax')
model = Model(input, network)
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Classic Neural Network Model Summary: ')
print(model.summary())

train_network(model, train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS)
results = model.evaluate(test_x, test_y)
c_accuracy = class_accuracy(model, test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
print(c_accuracy)
