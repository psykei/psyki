import distutils.cmd
from setuptools import setup, find_packages


class RunExperiments(distutils.cmd.Command):
    description = 'run injection experiments on poker hand dataset'
    user_options = [('experiments=', 'E', 'number of experiments'),
                    ('epochs=', 'e', 'number of epochs per experiment'),
                    ('layers=', 'l', 'number of layer of the neural network'),
                    ('neurons=', 'n', 'number of neurons per layer'),
                    ('batch=', 'b', 'batch size'),
                    ('knowledge=', 'k', 'use knowledge [Y/n]'),
                    ('prefix=', 'p', 'prefix of the experiment file that will be saved')]

    def initialize_options(self):
        self.experiments = 30
        self.epochs = 100
        self.layers = 3
        self.neurons = 128
        self.batch_size = 32
        self.knowledge = 'Y'
        self.prefix = ''

    def finalize_options(self):
        self.experiments = int(self.experiments)
        self.epochs = int(self.epochs)
        self.layers = int(self.layers)
        self.neurons = int(self.neurons)
        self.batch_size = int(self.batch_size)

    def run(self):
        from tensorflow.keras import Input, Model
        from tensorflow.keras.optimizers import Adam
        from test import get_mlp, train_network, get_processed_dataset
        from tensorflow.python.framework.random_seed import set_random_seed
        from antlr4 import CommonTokenStream, InputStream
        from psyki.ski import ConstrainingInjector
        from resources.dist.resources.DatalogLexer import DatalogLexer
        from resources.dist.resources.DatalogParser import DatalogParser
        from test import POKER_FEATURE_MAPPING, POKER_CLASS_MAPPING
        from test.resources import get_list_rules

        option_values = [self.experiments, self.epochs, self.layers, self.neurons, self.batch_size, self.knowledge, self.prefix]
        for i, option in enumerate(self.user_options):
            print(option[0] + str(option_values[i]))

        optimizer = Adam(learning_rate=0.001)
        train_x, train_y, test_x, test_y = get_processed_dataset('poker', validation=0.05)
        set_random_seed(123)

        for i in range(self.experiments):
            print('Experiment ' + str(i+1) + '/' + str(self.experiments))
            net_input = Input((10,), name='Input')
            network = get_mlp(net_input, output=10, layers=self.layers, neurons=self.neurons,
                              activation_function='relu',
                              last_activation_function='softmax')
            model = Model(net_input, network)
            if self.knowledge.lower() == 'y':
                file = self.prefix + '/model' + str(i + 1)
                rules = get_list_rules('poker-new')
                formulae = {rule: DatalogParser(CommonTokenStream(DatalogLexer(InputStream(rule)))) for rule in rules}
                injector = ConstrainingInjector(model, POKER_CLASS_MAPPING, POKER_FEATURE_MAPPING)
                injector.inject(formulae)
                injector.predictor.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model = injector.predictor

            else:
                file = self.prefix + '/model' + str(i + 1)
                model = Model(net_input, network)
                model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model with rules
            model.summary()
            train_network(model, train_x, train_y, test_x, test_y, batch_size=self.batch_size, epochs=self.epochs, file=file, knowledge=self.knowledge.lower() == 'y')


class TestAnalysis(distutils.cmd.Command):
    description = 'run evaluation of neural networks on test set'
    user_options = [('filename=', 'f', 'file name of the neural networks without the experiment number'),
                    ('min=', 'm', 'lowest experiment number (default = 1)'),
                    ('max=', 'M', 'greatest experiment number (default = 30)'),
                    ('save=', 's', 'name of the file with the results')
                    ]

    def initialize_options(self):
        self.filename = 'classic/model_L3_N128_E100_B32_I'
        self.min = 1
        self.max = 30
        self.save = 'dummy'

    def finalize_options(self):
        self.min = int(self.min)
        self.max = int(self.max)

    def run(self):
        import os
        from tensorflow.keras.models import load_model
        from tensorflow.keras.optimizers import Adam
        from test import class_accuracy, f1
        from test.experiments import models, statistics
        from test import get_processed_dataset

        option_values = [self.filename, self.min, self.max, self.save]
        for i, option in enumerate(self.user_options):
            print(option[0] + str(option_values[i]))

        _, _, test_x, test_y = get_processed_dataset('poker')
        info = ["model;acc;f1;classes"]
        optimizer = Adam(learning_rate=0.001)
        for i in range(self.min - 1, self.max):
            file_exp = self.filename + str(i + 1) + '.h5'
            file_exp = str(models.PATH / file_exp)
            print(file_exp)
            model = load_model(file_exp)
            model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            classes_accuracy = class_accuracy(model, test_x, test_y)
            macro_f1 = f1(model, test_x, test_y)
            info.append(
                os.path.basename(file_exp) + '; ' +
                str(model.evaluate(test_x, test_y)[1]) + '; ' +
                str(macro_f1) + '; ' +
                str(classes_accuracy)
            )
        with open(str(statistics.PATH / self.save) + '.csv', 'w') as f:
            for row in info:
                f.write("%s\n" % row)


setup(
    name='psyki',  # Required
    description='Platform for Symbolic Knowledge Injection',
    license='Apache 2.0 License',
    long_description_content_type='text/markdown',
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3 :: Only',
        'Programming Language :: Prolog'
    ],
    keywords='knowledge injection, symbolic ai, ski, extractor, rules',  # Optional
    # package_dir={'': 'src'},  # Optional
    packages=find_packages(),  # Required
    include_package_data=True,
    python_requires='>=3.9.0, <3.10',
    install_requires=[
        'tensorflow~=2.6.2',
        'scikit-learn~=1.0.1',
    ],  # Optional
    zip_safe=False,
    platforms="Independant",
    cmdclass={
        'run_experiments': RunExperiments,
        'run_test_evaluation': TestAnalysis
    },
)
