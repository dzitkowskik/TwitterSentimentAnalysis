# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
import inspect

__author__ = 'ghash'

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
from abc import ABCMeta, abstractmethod
from sklearn import cross_validation
import numpy as np
from TwitterSentimentAnalysis.datasets import TweetClassificationDatasetFactory
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.neuralnets import NNregression
from pybrain.tools.neuralnets import NNclassifier
import enum
import nltk
import pickle
import sklearn.linear_model as lm


@enum.unique
class AIEnum(enum.Enum):
    MultiClassClassificationNeuralNetwork = "Multi class classification NN"
    SimpleRegressionNeuralNetwork = "Simple regression NN"
    SimpleClassificationNeuralNetwork = "Simple classification NN"

    @classmethod
    def choices(cls):
        # get all members of the class
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        # filter down to just properties
        props = [m for m in members if not(m[0][:2] == '__')]
        # format into django choice tuple
        return tuple([(p[0], p[1].value) for p in props])


class NeuralNetwork(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def factory(ai_type):
        if ai_type == AIEnum.MultiClassClassificationNeuralNetwork:
            return MultiClassClassificationNeuralNetwork()
        if ai_type == AIEnum.SimpleClassificationNeuralNetwork:
            return SimpleClassificationNeuralNetwork()
        if ai_type == AIEnum.SimpleRegressionNeuralNetwork:
            return SimpleRegressionNeuralNetwork()
        assert 0, "Bad enum given: " + str(ai_type)

    @abstractmethod
    def run(self, ds_train, ds_test):
        pass

    @abstractmethod
    def test(self, ds_test):
        pass

    @abstractmethod
    def __call__(self, ds_train, ds_test):
        pass

    @abstractmethod
    def save(self, path):
        pass

    @abstractmethod
    def load(self, path):
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

    def test(self, ds_test):
        error = percentError(
            self.network.activateOnDataset(ds_test),
            ds_test['class'])

        print "Multi class neural network test error: %5.2f%%" % error
        return error

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

            ds_train = TweetClassificationDatasetFactory.convert_to_ds(x_train, y_train)
            ds_test = TweetClassificationDatasetFactory.convert_to_ds(x_test, y_test)
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

    def save(self, path):
        NetworkWriter.writeToFile(self.network, path)

    def load(self, path):
        self.network = NetworkReader.readFrom(path)


class SimpleRegressionNeuralNetwork(NeuralNetwork):
    def __init__(self, hid_cnt=10, convergence=0.01):
        self.hidden = hid_cnt
        self.network = None
        self.convergence = convergence

    def run(self, ds_train, ds_test):
        self.network = NNregression(ds_train)
        self.network.runTraining(self.convergence)
        tstresult = self.test(ds_test)
        return tstresult

    def test(self, ds_test):
        tstresult = percentError(
            self.network.activateOnDataset(ds_test),
            ds_test['class'])
        return tstresult

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

            ds_train = TweetClassificationDatasetFactory.convert_to_ds(x_train, y_train)
            ds_test = TweetClassificationDatasetFactory.convert_to_ds(x_test, y_test)

            self.network = NNregression(ds_train)

            self.network.runTraining(self.convergence)

            tstresult = self.test(ds_test)

            errors[i] = tstresult

            i += 1

        print "Simple Regression Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        NNregression.saveNetwork(self.network, path)

    def load(self, path):
        self.network = NetworkReader.readFrom(path)


class SimpleClassificationNeuralNetwork(NeuralNetwork):
    def __init__(self, hid_cnt=10, convergence=0.01):
        self.hidden = hid_cnt
        self.network = None
        self.convergence = convergence

    def run(self, ds_train, ds_test):
        self.network = NNclassifier(ds_train)
        self.network.runTraining(self.convergence)
        tstresult = self.test(ds_test)
        return tstresult

    def test(self, ds_test):
        tstresult = percentError(
            self.network.activateOnDataset(ds_test),
            ds_test['class'])
        return tstresult

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

            ds_train = TweetClassificationDatasetFactory.convert_to_ds(x_train, y_train)
            ds_test = TweetClassificationDatasetFactory.convert_to_ds(x_test, y_test)
            ds_train._convertToOneOfMany()
            ds_test._convertToOneOfMany()

            self.network = NNclassifier(ds_train)

            self.network.runTraining(self.convergence)

            tstresult = self.test(ds_test)

            errors[i] = tstresult

            i += 1

        print "Simple Classification Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        NNclassifier.saveNetwork(self.network, path)

    def load(self, path):
        self.network = NetworkReader.readFrom(path)


class NaiveBayesClassifier(NeuralNetwork):
    def __init__(self):
        self.classifier = nltk.NaiveBayesClassifier()

    def run(self, ds_train, ds_test):
        self.classifier.train(ds_train)
        tstresult = nltk.classify.accuracy(self.classifier, ds_test)
        return tstresult

    def test(self, ds_test):
        tstresult = nltk.classify.accuracy(self.classifier, ds_test)
        return tstresult

    def run_with_crossvalidation(self, ds, iterations=5):
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            train_ds = ds[train_index, :]
            test_ds = ds[test_index, :]

            self.classifier.train(train_ds)

            tstresult = nltk.classify.accuracy(self.classifier, test_ds)

            errors[i] = tstresult

            i += 1

        print "Naive Bayes Classifier cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load(self, path):
        f = open(path)
        self.classifier = pickle.load(f)
        f.close()


class MaxEntropyClassifier(NeuralNetwork):
    def __init__(self):
        self.classifier = nltk.MaxentClassifier()

    def run(self, ds_train, ds_test):
        self.classifier.train(ds_train)
        tstresult = nltk.classify.accuracy(self.classifier, ds_test)
        return tstresult

    def test(self, ds_test):
        tstresult = nltk.classify.accuracy(self.classifier, ds_test)
        return tstresult

    def run_with_crossvalidation(self, ds, iterations=5):
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            train_ds = ds[train_index, :]
            test_ds = ds[test_index, :]

            self.classifier.train(train_ds)

            tstresult = nltk.classify.accuracy(self.classifier, test_ds)

            errors[i] = tstresult

            i += 1

        print "Max Entropy Classifier cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load(self, path):
        f = open(path)
        self.classifier = pickle.load(f)
        f.close()


class LinearRegression(NeuralNetwork):
    def __init__(self):
        self.regression = lm.LinearRegression()

    def run(self, ds_train, ds_test):
        X_train = [train[0] for train in ds_train]
        y_train = [train[1] for train in ds_train]
        X_test = [test[0] for test in ds_test]
        y_test = [test[1] for test in ds_test]
        self.regression.fit(X_train, y_train)
        tstresult = self.regression.score(X_test, y_test)
        return tstresult

    def test(self, ds_test):
        X_test = [test[0] for test in ds_test]
        y_test = [test[1] for test in ds_test]
        tstresult = self.regression.score(X_test, y_test)
        return tstresult

    def run_with_crossvalidation(self, ds, iterations = 5):
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            train_ds = ds[train_index, :]
            test_ds = ds[test_index, :]

            X_train = [train[0] for train in train_ds]
            y_train = [train[1] for train in train_ds]
            X_test = [test[0] for test in test_ds]
            y_test = [test[1] for test in test_ds]
            
            self.regression.fit(X_train, y_train)
            errors[i] = self.regression.score(X_test, y_test)
            i += 1
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        f = open(path, 'wb')
        pickle.dump(self.regression, f)
        f.close()

    def load(self, path):
        f = open(path)
        self.regression = pickle.load(f)
        f.close()

