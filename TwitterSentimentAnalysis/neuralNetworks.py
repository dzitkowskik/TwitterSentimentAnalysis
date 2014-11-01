# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'ghash'

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
from abc import ABCMeta, abstractmethod
from sklearn import cross_validation
import numpy as np
from TwitterSentimentAnalysis.datasets import SimpleTweetDatasetFactory


class NeuralNetwork(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def run(self, ds_train, ds_test):
        pass

    @abstractmethod
    def __call__(self, ds_train, ds_test):
        pass


class MultiClassClassificationNeuralNetwork(NeuralNetwork):
    def __init__(self, inp_cnt, out_cnt, hid_cnt=10, epochs=100):
        self.hid_cnt = hid_cnt
        self.out_cnt = out_cnt
        self.inp_cnt = inp_cnt
        self.epochs = epochs
        self.network = self.__build_default_network()

    def apply_custom_network(self, hidden_counts):
        network = FeedForwardNetwork()
        in_layer = LinearLayer(self.inp_cnt)
        network.addInputModule(in_layer)
        out_layer = SoftmaxLayer(self.out_cnt)
        network.addOutputModule(out_layer)

        hidden_layer = SigmoidLayer(hidden_counts[0])
        network.addModule(hidden_layer)
        in_to_hidden = FullConnection(in_layer, hidden_layer)
        network.addConnection(in_to_hidden)

        for i in range(1, len(hidden_counts)):
            last_hidden_layer = hidden_layer
            hidden_layer = SigmoidLayer(hidden_counts[i])
            network.addModule(hidden_layer)
            hidden_to_hidden = FullConnection(last_hidden_layer, hidden_layer)
            network.addConnection(hidden_to_hidden)

        hidden_to_out = FullConnection(hidden_layer, out_layer)
        network.addConnection(hidden_to_out)
        network.sortModules()
        self.network = network
        return self

    def __build_default_network(self):
        return buildNetwork(self.inp_cnt, self.hid_cnt, self.out_cnt, outclass=SoftmaxLayer, bias=True)

    # noinspection PyProtectedMember
    def run(self, ds_train, ds_test):
        ds_train._convertToOneOfMany()
        ds_test._convertToOneOfMany()

        trainer = BackpropTrainer(
            self.network,
            dataset=ds_train,
            momentum=0.1,
            verbose=True,
            weightdecay=0.01)

        trainer.trainEpochs(self.epochs)
        tstresult = percentError(
            trainer.testOnClassData(dataset=ds_test),
            ds_test['class'])

        print "Multi class neural network test error: %5.2f%%" % tstresult
        return tstresult

    # noinspection PyProtectedMember
    def run_with_crossvalidation(self, ds, iterations=5):
        x = ds['input']
        y = ds['target']

        n, m = x.shape
        errors = np.zeros(iterations)

        cv = cross_validation.KFold(n, iterations, shuffle=True)

        i = 0
        for train_index, test_index in cv:
            x_train = x[train_index, :]
            y_train = y[train_index, :]
            x_test = x[test_index, :]
            y_test = y[test_index, :]

            ds_train = SimpleTweetDatasetFactory.convert_to_ds(x_train, y_train)
            ds_test = SimpleTweetDatasetFactory.convert_to_ds(x_test, y_test)
            ds_train._convertToOneOfMany()
            ds_test._convertToOneOfMany()

            trainer = BackpropTrainer(
                self.network,
                dataset=ds_train,
                momentum=0.1,
                verbose=True,
                weightdecay=0.01)

            trainer.trainEpochs(self.epochs)
            errors[i] = percentError(
                trainer.testOnClassData(dataset=ds_test),
                ds_test['class'])
            i += 1

        print "Multi class NN cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)
