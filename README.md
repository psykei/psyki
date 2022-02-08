# PSyKI

## Intro

PSyKI (Platform for Symbolic Knowledge Injection) is intended to be a library for injection symbolic knowledge into sub-symbolic predictors.
At the moment PSyKe offers a FOL interpreter for rules expressed in Skolem normal form and a general injection techniques for neural networks (NN) based on structuring.

An `Injector` is an object capable of injecting symbolic knowledge into a NN with method `inject`.
A `StructuringInjector` embeds textual FOL rules into network modules/sub-networks capable of evaluating the truth degree of the formulae themselves.

User must provide in addition to the textual rule a mapping between input variables and input features.
Variables' names and their indices correspond to vector input position of the NN.
For example, network predicts iris class from 4 ordered features: sepal length (SL), sepal width (SW), petal length (PL) and petal width (PW).
In FOL formulae variable names SL, SW, PL and PW can be properly used if user defines the following mapping:
```python
features_mapping = {
    'SL': 0,
    'SW': 1,
    'PL': 2,
    'PW': 3,
}
```

## Users

### Requirements

- python 3.9+
- tensorflow 2.6.2+
- scikit-learn 1.0.1+
- numpy 1.21.4+
- pandas 1.3.4+
- scipy 1.7.1+

### Rule convention
Ascii symbols for logic operators:
- variable: camel case string (for scalars defined in variables mapping), X (for vector/class prediction)
- constant: number (for scalars), lower case string (for vector/class representation defined in output mapping)
- equivalence: = (for scalar variables and constants), |= (for vectors)
- not: ~
- conjunction: ^
- disjunction: ∨
- if ... then: ->
- ... if: <-
- if and only if: <->
- not equal: !=
- less: <
- less equal: <=
- greater: >
- greater equal: >=
- (math) plus: +
- (math) times: *

Note: to omit a rule for a class (constant cost 0) it is sufficient to use the skip simbol " / ".

Example for iris:

```text
PL <= 2.28 <- X |= setosa
PL > 2.28 ^ PW <= 1.64 <- X |= versicolor
PL > 2.28 ^ PW > 1.64 <- X |= virginica
```

### Demo

`demo_structuring.ipynb` is a notebook that shows how injection via structuring is applied to a network for iris classification task.
Rules are defined in `resources/rules/iris.csv`.


Example of injection:
```python
input_features = Input((4,), name='Input')
injector = StructuringInjector(parser)
network = get_mlp(input=input_features, output=3, layers=3, neurons=16, activation_function='relu',
                  last_activation_function='softmax')
main_network = Model(input_features, network).layers[-2].output
model = injector.inject(iris_rules, input_features, main_network, 3, 'softmax', features_mapping)
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.summary()
```

Output:
```text
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input (InputLayer)              [(None, 4)]          0                                            
__________________________________________________________________________________________________
lambda_4 (Lambda)               (None, 1)            0           Input[0][0]                      
__________________________________________________________________________________________________
dense_10 (Dense)                (None, 1)            5           Input[0][0]                      
__________________________________________________________________________________________________
lambda (Lambda)                 (None, 1)            0           Input[0][0]                      
__________________________________________________________________________________________________
dense (Dense)                   (None, 1)            5           Input[0][0]                      
__________________________________________________________________________________________________
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
__________________________________________________________________________________________________
L_2 (Dense)                     (None, 16)           272         L_1[0][0]                        
__________________________________________________________________________________________________
maximum (Maximum)               (None, 1)            0           dense_1[0][0]                    
                                                                 dense_2[0][0]                    
__________________________________________________________________________________________________
minimum (Minimum)               (None, 1)            0           dense_9[0][0]                    
                                                                 maximum_2[0][0]                  
__________________________________________________________________________________________________
minimum_1 (Minimum)             (None, 1)            0           dense_18[0][0]                   
                                                                 dense_20[0][0]                   
__________________________________________________________________________________________________
concatenate_11 (Concatenate)    (None, 19)           0           L_2[0][0]                        
                                                                 maximum[0][0]                    
                                                                 minimum[0][0]                    
                                                                 minimum_1[0][0]                  
__________________________________________________________________________________________________
dense_21 (Dense)                (None, 3)            60          concatenate_11[0][0]             
==================================================================================================
Total params: 458
Trainable params: 412
Non-trainable params: 46
```

### Experiments

Command `python setup.py run_experiments_structuring` trains neural networks to classify poker hands using injection of FOL rules.
Run `python setup.py --help run_experiments_structuring` for more details.
Rules are provided in `resources/rules/poker.csv`.
The usage of rules raises execution time w.r.t. the same network configuration.
With:
- one rule (some of them quite articulated) for each class;
- a fully connected 3-layers NN of 128 neurons per layer;
- a training set of 25.010 records;
- a validation set of 50.000 records;
- an Apple M1 with 8-core CPU, 8-core GPU and 8 GB of RAM;

execution time for one epoch is approximately 2-4 seconds depending on the number of injected rules.

`test/experiments/models` contains the models of neural networks trained after the execution of the script with the following hyperparameters:
- layers = 3
- neurons = 128
- epochs = 100
- batch size = 32
- experiments = 30

Expected output:
```text
running run_experiments_structuring
experiments=30
epochs=100
layers=3
neurons=128
batch=32
knowledge=Y
file=dummy
rules=10
Experiment 1/30
Model: "model_1"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input (InputLayer)              [(None, 10)]         0                                            
__________________________________________________________________________________________________
dense_1170 (Dense)              (None, 1)            11          Input[0][0]                      
__________________________________________________________________________________________________
lambda_1255 (Lambda)            (None, 1)            0           Input[0][0]                      
__________________________________________________________________________________________________
_ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ _ 
__________________________________________________________________________________________________
dense_5434 (Dense)              (None, 10)           1390        concatenate_3916[0][0]           
==================================================================================================
Total params: 21,532
Trainable params: 19,310
Non-trainable params: 2,222
__________________________________________________________________________________________________
```

```text
Epoch 1/100
782/782 [==============================] - 19s 14ms/step - loss: 0.6342 - accuracy: 0.8866 - val_loss: 0.4306 - val_accuracy: 0.9237
Epoch 2/100
782/782 [==============================] - 9s 11ms/step - loss: 0.3253 - accuracy: 0.9287 - val_loss: 0.2458 - val_accuracy: 0.9459
Epoch 3/100
782/782 [==============================] - 9s 11ms/step - loss: 0.1961 - accuracy: 0.9561 - val_loss: 0.1555 - val_accuracy: 0.9601
Epoch 4/100
782/782 [==============================] - 9s 11ms/step - loss: 0.1264 - accuracy: 0.9805 - val_loss: 0.1024 - val_accuracy: 0.9836
Epoch 5/100
532/782 [===================>..........] - ETA: 1s - loss: 0.0900 - accuracy: 0.9902interrupted
```

Modes are divided by the knowledge injection received:
- `test/experiments/models/classic` = no injection
- `test/experiments/models/structuring[1-10]` = injection of the specified number of rules

Training history of each network is stored in `test/experiments/statistics/classic`, `test/experiments/statistics/structuring[1-10]`.
Test evaluations are saved in `test/experiments/statistics/`.

To run the evaluation of the networks on the test set execute command `python setup.py run_test_evaluation`.
Run python `setup.py --help run_test_evaluation` for more details.

Expected output:
```text
running run_test_evaluation
filename=structuring7/model
min=1
max=30
save=dummy
/Users/matteomagnini/Desktop/Repo/psyki/test/experiments/models/structuring7/model_0.h5
31250/31250 [==============================] - 54s 2ms/step - loss: 0.0133 - accuracy: 0.9993
/Users/matteomagnini/Desktop/Repo/psyki/test/experiments/models/structuring7/model_1.h5
31250/31250 [==============================] - 51s 2ms/step - loss: 0.0295 - accuracy: 0.9954
```

