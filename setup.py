import distutils.cmd
from setuptools import setup, find_packages
from psyki.logic import Parser


class RunExperiments(distutils.cmd.Command):
    description = 'run injection experiments on poker hand dataset'
    user_options = [('experiments=', 'E', 'number of experiments'),
                    ('epochs=', 'e', 'number of epochs per experiment'),
                    ('layers=', 'l', 'number of layer of the neural network'),
                    ('neurons=', 'n', 'number of neurons per layer'),
                    ('batch=', 'b', 'batch size'),
                    ('knowledge=', 'k', 'use knowledge [Y/n]'),
                    ('file=', 'f', 'name of the experiment file that will be saved'),
                    ('rules=', 'r', 'number of rules to inject')]

    def initialize_options(self):
        self.experiments = 30
        self.epochs = 100
        self.layers = 3
        self.neurons = 128
        self.batch_size = 32
        self.knowledge = 'Y'
        self.file = ''
        self.rules = 10

    def finalize_options(self):
        self.experiments = int(self.experiments)
        self.epochs = int(self.epochs)
        self.layers = int(self.layers)
        self.neurons = int(self.neurons)
        self.batch_size = int(self.batch_size)
        self.rules = int(self.rules)

    def run(self):
        option_values = [self.experiments, self.epochs, self.layers, self.neurons, self.batch_size, self.knowledge,
                         self.file, self.rules]

        for i, option in enumerate(self.user_options):
            print(option[0] + str(option_values[i]))

    def iteration_variables(self, i: int):
        from tensorflow.keras import Input
        from test import get_mlp

        print('Experiment ' + str(i + 1) + '/' + str(self.experiments))
        net_input = Input((10,), name='Input')
        network = get_mlp(net_input, output=10, layers=self.layers, neurons=self.neurons,
                          hidden_activation='relu',
                          last_activation='softmax')
        file = self.file + '/model' + str(i + 1)
        return net_input, network, file


class RunExperimentsStructuring(RunExperiments):
    description = 'run injection with structuring1 experiments on poker hand dataset'

    def run(self):
        from tensorflow.keras import Model
        from tensorflow.keras.optimizers import Adam
        from test import train_network, get_processed_dataset
        from test import POKER_INPUT_MAPPING
        from test import POKER_RULES
        from psyki import StructuringInjector

        super().run()

        optimizer = Adam()
        train_x, train_y, test_x, test_y = get_processed_dataset('poker', validation=0.05)

        for i in range(self.experiments):
            net_input, network, file = self.iteration_variables(i)

            if self.knowledge.lower() == 'y':
                main_network = Model(net_input, network).layers[-2].output
                injector = StructuringInjector(Parser.default_parser())
                model = injector.inject(POKER_RULES, net_input, main_network, 10, 'softmax', POKER_INPUT_MAPPING)
            else:
                model = Model(net_input, network)

            model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            model.summary()
            train_network(model, train_x, train_y, test_x, test_y, self.batch_size, epochs=self.epochs, file=file)


class TestAnalysis(distutils.cmd.Command):
    description = 'run evaluation of neural networks on test set'
    user_options = [('filename=', 'f', 'file name of the neural networks without the experiment number'),
                    ('min=', 'm', 'lowest experiment number (default = 1)'),
                    ('max=', 'M', 'greatest experiment number (default = 30)'),
                    ('save=', 's', 'name of the file with the results')
                    ]

    def initialize_options(self):
        self.filename = 'structuring1/model'
        self.min = 1
        self.max = 30
        self.save = 'test_results_stucturing'

    def finalize_options(self):
        self.min = int(self.min)
        self.max = int(self.max)

    def run(self):
        import os
        from tensorflow.keras.optimizers import Adam
        from test import class_accuracy, f1
        from test import get_processed_dataset
        from psyki import StructuringInjector

        option_values = [self.filename, self.min, self.max, self.save]
        for i, option in enumerate(self.user_options):
            print(option[0] + str(option_values[i]))

        _, _, test_x, test_y = get_processed_dataset('poker')
        info = ["model;acc;f1;classes"]
        optimizer = Adam(learning_rate=0.001)
        for i in range(self.min - 1, self.max):
            file_exp = self.filename + '_' + str(i) + '.h5'
            print(file_exp)
            model = StructuringInjector.load_model(file_exp)
            model.compile(optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            classes_accuracy = class_accuracy(model, test_x, test_y)
            macro_f1 = f1(model, test_x, test_y)
            info.append(
                os.path.basename(file_exp) + '; ' +
                str(model.evaluate(test_x, test_y)[1]) + '; ' +
                str(macro_f1) + '; ' +
                str(classes_accuracy)
            )
        with open(self.save + '.csv', 'w') as f:
            for row in info:
                f.write("%s\n" % row)


class ASTVisualizer(distutils.cmd.Command):
    description = 'Print the ast of a rule'
    user_options = [('file=', 'f', 'file name of the rules'),
                    ('rule=', 'r', 'name of the rule'),
                    ('flat=', 'f', 'flat ast'),
                    ]

    def initialize_options(self):
        self.filename = 'poker'
        self.rule = 'pair'
        self.flat = False

    def finalize_options(self):
        self.flat = self.flat == 'True'

    def run(self):
        from test import get_rules

        rules = get_rules(self.filename)
        rule = rules[self.rule]
        parser = Parser.default_parser()
        print(parser.structure(rule, self.flat, True))


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
        'run_experiments_structuring': RunExperimentsStructuring,
        'run_test_evaluation': TestAnalysis,
        'run_ast_visualizer': ASTVisualizer
    },
)
