# PSyKI

## Intro

PSyKI (Platform for Symbolic Knowledge Injection) is intended to be a library for injection symbolic knowledge into sub-symbolic predictors.
At the moment PSyKe offers a FOL interpreter for rules expressed in Skolem normal form and a general injection technique for neural networks (NN).

An `Injector` is an object capable of provide rules to a NN with method `inject`.
Rules are textual FOL rules that are processed into fuzzy logic functions.
To convert textual rules user can use method `get_function` of class `Parser`.

User must provide in addition to the textual rule two mappings:

- variables' names and their indices (corresponding to vector input position of the NN).
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
- output class mapping.
For example, with 3 classes and one-hot encoding:
```python
class_mapping = {
  'setosa': tf.constant([1., 0., 0.]),
  'virginica': tf.constant([0., 1., 0.]),
  'versicolor': tf.constant([0., 0., 1.])
}
```

## Users

### Requirements

- python 3.9+
- tensorflow 2.6.2+
- scikit-learn 1.0.1+
- numpy 1.21.4+
- keras 2.7.0+
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
PL > 2.28 ∨ PW > 1.64 <- X |= virginica
PL > 2.28 ∨ PW <= 1.64 <- X |= versicolor
```

### Demo

`demo.ipynb` is a notebook that shows how injection is applied to a network for iris classification task.
Rules are defined in `resources/rules/iris.csv`.


Example of injection:
```python
parser = Parser.default_parser()
iris_rules = [parser.get_function(rule, features_mapping, class_mapping)
               for _, rule in get_rules('iris').items()]
input_features = Input((4,), name='Input')
network = get_mlp(input=input_features, output=3, layers=3, neurons=32, activation_function='relu',
                  last_activation_function='softmax')
injector = Injector(network, input_features)
injector.inject(iris_rules)
new_network = injector.predictor
new_network.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])
new_network.summary()
```

Output:
```text
Model: "model"
__________________________________________________________________________________________________
Layer (type)                    Output Shape         Param #     Connected to                     
==================================================================================================
Input (InputLayer)              [(None, 4)]          0                                            
__________________________________________________________________________________________________
L_1 (Dense)                     (None, 32)           160         Input[0][0]                      
__________________________________________________________________________________________________
L_2 (Dense)                     (None, 32)           1056        L_1[0][0]                        
__________________________________________________________________________________________________
L_3 (Dense)                     (None, 3)            99          L_2[0][0]                        
__________________________________________________________________________________________________
Concatenate (Concatenate)       (None, 7)            0           Input[0][0]                      
                                                                 L_3[0][0]                        
__________________________________________________________________________________________________
Knowledge (Lambda)              (None, 3)            0           Concatenate[0][0]                
==================================================================================================
Total params: 1,315
Trainable params: 1,315
Non-trainable params: 0
```

### Experiments

Command `python setup.py run_experiments` trains neural networks to classify poker hands using injection of FOL rules.
Run `python setup.py --help run_experiments` for more details.
Rules are provided in `resources/rules/poker.csv`.
The usage of rules raises execution time w.r.t. the same network configuration.
With:
- one rule (some of them quite articulated) for each class;
- a fully connected 3-layers NN of 128 neurons per layer;
- a training set of 25.010 records;
- a validation set of 50.000 records;
- an Apple M1 with 8-core CPU, 8-core GPU and 8 GB of RAM;

execution time for one epoch is approximately 20 seconds.

`test/experiments/models` contains the models of neural networks trained after the execution of the script with the following hyperparameters:
- layers = 3
- neurons = 128
- epochs = 100
- batch size = 32
- experiments = 30

Modes are divided by the knowledge injection received:
- `test/experiments/models/classic` = no injection
- `test/experiments/models/R0` = one rule for each class
- `test/experiments/models/R1` = one rule for class nothing
- `test/experiments/models/R2` = one rule for classes nothing and pair

Training history of each network is stored in `test/experiments/statistics/classic`, `test/experiments/statistics/R0`, `test/experiments/statistics/R1` and `test/experiments/statistics/R2`.
Test evaluations are saved in `test/experiments/statistics/`.

