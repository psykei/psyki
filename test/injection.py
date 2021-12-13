import numpy as np
from sklearn.preprocessing import OneHotEncoder
from tensorflow.python.framework.ops import disable_eager_execution
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from psyki import Injector
from psyki.fol import Parser
from psyki.fol.ast import AST
from psyki.fol.operators import *
from test.resources import get_dataset, get_rules

SEED: int = 123
EPOCHS: int = 200
NEURONS: int = 20

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
rules = []
for textual_rule in textual_rules:
    parser = Parser([L, LTX, LTY, LTEquivalence, Equivalence, Conjunction, Implication, LeftPar, RightPar, Exist,
                     Disjunction, Plus, Negation, Numeric, Product])
    parsed_rule = parser.parse(textual_rule)
    ast = AST()
    for operator in parsed_rule:
        ast.insert(operator[0], operator[1])
    rules.append(ast.root.call(input_mapping, output_mapping))


# Build the model

disable_eager_execution()


def get_injector():
    optimizer = Adam(learning_rate=0.001)
    model = Sequential()
    model.add(Dense(NEURONS, input_shape=(10,), activation='relu', name='input_layer'))
    model.add(Dense(NEURONS, activation='relu', name='hidden_layer'))
    model.add(Dense(10, activation='softmax', name='output_layer'))
    print('Neural Network Model Summary: ')
    print(model.summary())
    injector = Injector(model)
    injector.inject(rules)
    injector.predictor.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return injector


# Train the model with rules
injector = get_injector()
injector.predictor.fit(train_x, train_y, verbose=2, batch_size=5, epochs=EPOCHS)


def class_accuracy(_model, _x, _y):
    predicted_y = np.argmax(_model.predict(_x), axis=1)
    match = np.equal(predicted_y, _y)
    accuracy = []
    for i in range(10):
        accuracy.append([sum(match[_y == i]) / sum(_y == i), sum(_y == i)])
    return accuracy


# Disable rules
injector.knowledge = False
results = injector.predictor.evaluate(test_x, test_y)
c_accuracy = class_accuracy(injector.predictor, test_x, test_y)

print('Final test set loss after removing rules: {:4f}'.format(results[0]))
print('Final test set accuracy after removing rules: {:4f}'.format(results[1]))
print(c_accuracy)