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
  - SL -> 0
  - SW -> 1
  - PL -> 2
  - PW -> 3

- output class mapping.
For example, with 3 classes and one-hot encoding:
  - setosa -> [1, 0, 0] 
  - versicolor -> [0, 1, 0]
  - virginica -> [0, 0, 1]

Note: to omit a rule for a class (constant cost 0) it is sufficient to use the skip simbol " / ".

## Users

### Requirements

- python 3.9+
- tensorflow 2.6.2+
- scikit-learn 1.0.1+
- numpy 1.21.4+
- keras 2.7.0+
- pandas 1.3.4+
- scipy 1.7.1+

### Demo

`demo.ipynb` is a notebook that shows how injection is applied to a network for iris classification task.
Rules are defined in `resources/rules/iris.csv`.

### Experiments

Script `test/experiments/injection.py` trains neural networks to classify poker hands using injection of FOL rules.
Rules are provided in `resources/rules/poker.csv`.
The usage of rules raises execution time w.r.t. the same network configuration.
With:
- one rule (some of them quite articulated) for each class;
- a fully connected 3-layers NN of 128 neurons per layer;
- a training set of 25.010 records;
- a validation set of 50.000 records;
- an Apple M1 with 8-core CPU, 8-core GPU and 8 GB of RAM;

execution time for one epoch is approximately 20 seconds.

