import numpy as np
from tensorflow.keras import Input
from tensorflow.keras.activations import softmax
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework.ops import disable_eager_execution
from keras.optimizer_v2.adam import Adam
from psyki import Injector
from psyki.fol import Parser
from psyki.fol.operators import *
from test.resources import get_dataset, get_rules

SEED: int = 123
EPOCHS: int = 50
NEURONS: int = 20
BATCH_SIZE: int = 5

# disable_eager_execution()

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

textual_rules = get_rules('poker')
input_mapping = {
        'S1': 0,
        'R1': 1,
        'S2': 2,
        'R2': 3,
        'S3': 4,
        'R3': 5,
        'S4': 6,
        'R4': 7,
        'S5': 8,
        'R5': 9
    }
output_mapping = {
        'nothing':          tf.constant([1, 0, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'pair':             tf.constant([0, 1, 0, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'twoPairs':         tf.constant([0, 0, 1, 0, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'tris':             tf.constant([0, 0, 0, 1, 0, 0, 0, 0, 0, 0], dtype=tf.float32),
        'straight':         tf.constant([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=tf.float32),
        'flush':            tf.constant([0, 0, 0, 0, 0, 1, 0, 0, 0, 0], dtype=tf.float32),
        'full':             tf.constant([0, 0, 0, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32),
        'poker':            tf.constant([0, 0, 0, 0, 0, 0, 0, 1, 0, 0], dtype=tf.float32),
        'straightFlush':    tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 1, 0], dtype=tf.float32),
        'royalFlush':       tf.constant([0, 0, 0, 0, 0, 0, 0, 0, 0, 1], dtype=tf.float32)
    }
parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, ReverseImplication, LeftPar, RightPar,
                 Exist, Disjunction, Plus, Negation, Numeric, Product, Disequal, DoubleImplication, LessEqual])
rules = [parser.get_function(rule, input_mapping, output_mapping) for _, rule in textual_rules.items()]

# Build the model
optimizer = Adam(learning_rate=0.001)


def get_injector():
    input = Input((10,))
    x = Dense(NEURONS, activation='relu', name='input_layer')(input)
    x = Dense(NEURONS, activation='relu', name='hidden_layer')(x)
    x = Dense(10, activation='softmax', name='output_layer')(x)
    injector = Injector(x, input, softmax)
    injector.inject(rules)
    injector.predictor.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    print('Neural Network With Knowledge Model Summary: ')
    print(injector.predictor.summary())
    # plot_model(injector.predictor)
    return injector


def class_accuracy(_model, _x, _y):
    predicted_y = np.argmax(_model.predict(_x), axis=1)
    match = np.equal(predicted_y, _y)
    accuracy = []
    for i in range(10):
        accuracy.append([sum(match[_y == i]) / sum(_y == i), sum(_y == i)])
    return accuracy


# Train the model with rules
injector = get_injector()
injector.predictor.fit(train_x, train_y, verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS)

# Disable rules
injector.knowledge = False
results = injector.predictor.evaluate(test_x, test_y)
c_accuracy = class_accuracy(injector.predictor, test_x, test_y)

print('Final test set loss after removing rules: {:4f}'.format(results[0]))
print('Final test set accuracy after removing rules: {:4f}'.format(results[1]))
print(c_accuracy)


model = None
model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
print('Classic Neural Network Model Summary: ')
print(injector.predictor.summary())

model.fit(train_x, train_y, verbose=1, batch_size=BATCH_SIZE, epochs=EPOCHS)
results = injector.predictor.evaluate(test_x, test_y)
c_accuracy = class_accuracy(injector.predictor, test_x, test_y)

print('Final test set loss: {:4f}'.format(results[0]))
print('Final test set accuracy: {:4f}'.format(results[1]))
print(c_accuracy)


