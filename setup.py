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
        pass

    def run(self):
        from keras import Model
        from tensorflow.keras import Input
        from tensorflow.keras.activations import softmax
        from keras.optimizer_v2.adam import Adam
        from psyki import Injector
        from test import POKER_RULES, get_mlp, train_network, get_processed_dataset

        option_values = [self.experiments, self.epochs, self.layers, self.neurons, self.batch_size, self.knowledge, self.prefix]
        for i, option in enumerate(self.user_options):
            print(option[0] + str(option_values[i]))

        optimizer = Adam(learning_rate=0.001)
        train_x, train_y, test_x, test_y = get_processed_dataset('poker', validation=0.05)

        for i in range(self.experiments):
            print('Experiment ' + str(i+1) + '/' + str(self.experiments))
            net_input = Input((10,), name='Input')
            network = get_mlp(net_input, output=10, layers=self.layers, neurons=self.neurons,
                              activation_function='relu',
                              last_activation_function='softmax')
            main_file_name = str(self.layers) + '_N' + str(self.neurons) + '_E' + str(self.epochs) + '_B' + \
                             str(self.batch_size) + '_I' + str(i + 1)
            if self.knowledge.lower() == 'y':
                file = self.prefix + 'injection_L' + main_file_name
                injector = Injector(network, net_input, softmax)
                injector.inject(POKER_RULES)
                injector.predictor.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model = injector.predictor

            else:
                file = self.prefix + 'classic_L' + main_file_name
                model = Model(net_input, network)
                model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

            # Train the model with rules
            model.summary()
            train_network(model, train_x, train_y, test_x, test_y, batch_size=self.batch_size, epochs=self.epochs, file=file)

            # Save the base network without knowledge layer
            if self.knowledge.lower() == 'y':
                model = Model(inputs=model.net_input, outputs=model.layers[-3].output)
                model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
                model.save(file)


class TestAnalysis(distutils.cmd.Command):
    description = 'run evaluation of neural networks on test set'
    user_options = [('filename=', 'f', 'file name of the neural networks without the experiment number'),
                    ('min=', 'm', 'lowest experiment number (default = 1)'),
                    ('max=', 'M', 'greatest experiment number (default = 30)'),
                    ('save=', 's', 'name of the file with the results')
                    ]

    def initialize_options(self):
        self.filename = 'R2/injection_L3_N128_E100_B32'
        self.min = 1
        self.max = 30
        self.save = 'test_results'

    def finalize_options(self):
        pass

    def run(self):
        import os
        from keras.models import load_model
        from keras.optimizer_v2.adam import Adam
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
            file_exp = self.filename + '_I' + str(i + 1) + '.h5'
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