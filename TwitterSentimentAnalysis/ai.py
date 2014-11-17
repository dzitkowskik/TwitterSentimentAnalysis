# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
from abc import ABCMeta, abstractmethod
from sklearn import cross_validation
import numpy as np
from TwitterSentimentAnalysis.datasets import TweetClassificationDatasetFactory, ProblemTypeEnum
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.neuralnets import NNregression
from pybrain.tools.neuralnets import NNclassifier
import enum
import nltk
import pickle
import sklearn.linear_model as lm
import inspect


@enum.unique
class AIEnum(enum.Enum):
    MultiClassClassificationNeuralNetwork = "Multi class classification NN"
    SimpleRegressionNeuralNetwork = "Simple regression NN"
    SimpleClassificationNeuralNetwork = "Simple classification NN"
    NaiveBayesClassifier = "Naive Bayes classifier"
    MaxEntropyClassifier = "Max entropy classifier"
    LinearRegression = "Linear regression"

    @classmethod
    def choices(cls):
        # get all members of the class
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        # filter down to just properties
        props = [m for m in members if not(m[0][:2] == '__')]
        # format into django choice tuple
        return tuple([(p[0], p[1].value) for p in props])


class AI(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def factory(ai_type):
        if ai_type == AIEnum.MultiClassClassificationNeuralNetwork:
            return MultiClassClassificationNeuralNetwork()
        elif ai_type == AIEnum.SimpleClassificationNeuralNetwork:
            return SimpleClassificationNeuralNetwork()
        elif ai_type == AIEnum.SimpleRegressionNeuralNetwork:
            return SimpleRegressionNeuralNetwork()
        elif ai_type == AIEnum.NaiveBayesClassifier:
            return NaiveBayesClassifier()
        elif ai_type == AIEnum.MaxEntropyClassifier:
            return MaxEntropyClassifier()
        elif ai_type == AIEnum.LinearRegression:
            return LinearRegression()
        assert 0, "Bad enum given: " + str(ai_type)

    @staticmethod
    def load_network_from_file(file_path):
        # TODO: Implement loading suitable AI from file
        # !! It requires a knowledge of what type of AI it is !!
        pass

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

    @abstractmethod
    def get_type(self):
        pass

    @abstractmethod
    def fill_with_predicted_data(self, ds, data):
        pass


class MultiClassClassificationNeuralNetwork(AI):
    def __init__(self, inp_cnt=3, out_cnt=9, hid_cnt=10, epochs=100):
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
        result = self.network.activateOnDataset(ds_test)
        error = percentError(result, ds_test['class'])
        print "Multi class neural network test error: %5.2f%%" % error
        return error, result

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

    def get_type(self):
        return ProblemTypeEnum.Classification

    def fill_with_predicted_data(self, ds, data):
        out = self.network.activateOnDataset(ds)
        results = np.ravel(np.argmax(out, 1))
        i = 0
        assert(len(ds) == len(data))
        for record in data:
            record['sentiment'] = ds['target'][i]
            record['predicted_sentiment'] = results[i]
            i += 1


class SimpleRegressionNeuralNetwork(AI):
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
        result = self.network.activateOnDataset(ds_test)
        error = percentError(result, ds_test['class'])
        return error, result

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

    def get_type(self):
        return ProblemTypeEnum.Regression

    def fill_with_predicted_data(self, ds, data):
        # TODO: Implement filling data (list of dictionaries)
        # with predicted values on ds dataset
        pass


class SimpleClassificationNeuralNetwork(AI):
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
        result = self.network.activateOnDataset(ds_test)
        error = percentError(result, ds_test['class'])
        return error, result

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

    def get_type(self):
        return ProblemTypeEnum.Classification

    def fill_with_predicted_data(self, ds, data):
        # TODO: Implement filling data (list of dictionaries)
        # with predicted values on ds dataset
        pass


class NaiveBayesClassifier(AI):
    def __init__(self):
        self.classifier = None

    def run(self, ds_train, ds_test):
        X_train = ds_train['input']
        y_train = ds_train['class']
        X_test = ds_test['input']
        y_test = ds_test['class']

        train_fs = [(X_train[i], y_train[i]) for i in enumerate(X_train)]
        test_fs = [(X_test[i], y_test[i]) for i in enumerate(X_test)]

        self.classifier = nltk.NaiveBayesClassifier().train(train_fs)

        self.classifier.train(train_fs)
        tstresult = nltk.classify.accuracy(self.classifier, test_fs)
        return tstresult

    def test(self, ds_test):
        X_test = ds_test['input']
        y_test = ds_test['class']
        test_fs = [(X_test[i], y_test[i]) for i in enumerate(X_test)]
        tstresult = nltk.classify.accuracy(self.classifier, test_fs)
        return tstresult

    def run_with_crossvalidation(self, ds, iterations=5):
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            ds_train = ds[train_index, :]
            ds_test = ds[test_index, :]

            X_train = ds_train['input']
            y_train = ds_train['class']
            X_test = ds_test['input']
            y_test = ds_test['class']

            train_fs = [(X_train[i], y_train[i]) for i in enumerate(X_train)]
            test_fs = [(X_test[i], y_test[i]) for i in enumerate(X_test)]

            self.classifier.train(train_fs)

            tstresult = nltk.classify.accuracy(self.classifier, test_fs)

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

    def get_type(self):
        return ProblemTypeEnum.Classification

    def fill_with_predicted_data(self, ds, data):
        # TODO: Implement filling data (list of dictionaries)
        # with predicted values on ds dataset
        pass


class MaxEntropyClassifier(AI):
    def __init__(self):
        self.classifier = nltk.MaxentClassifier()

    def run(self, ds_train, ds_test):
        X_train = ds_train['input']
        y_train = ds_train['class']
        X_test = ds_test['input']
        y_test = ds_test['class']

        train_fs = [(X_train[i], y_train[i]) for i in enumerate(X_train)]
        test_fs = [(X_test[i], y_test[i]) for i in enumerate(X_test)]

        self.classifier.train(train_fs)
        tstresult = nltk.classify.accuracy(self.classifier, test_fs)
        return tstresult

    def test(self, ds_test):
        X_test = ds_test['input']
        y_test = ds_test['class']
        test_fs = [(X_test[i], y_test[i]) for i in enumerate(X_test)]
        tstresult = nltk.classify.accuracy(self.classifier, test_fs)
        return tstresult

    def run_with_crossvalidation(self, ds, iterations=5):
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            ds_train = ds[train_index, :]
            ds_test = ds[test_index, :]
            X_train = ds_train['input']
            y_train = ds_train['class']
            X_test = ds_test['input']
            y_test = ds_test['class']

            train_fs = [(X_train[i], y_train[i]) for i in enumerate(X_train)]
            test_fs = [(X_test[i], y_test[i]) for i in enumerate(X_test)]

            self.classifier.train(train_fs)
            tstresult = nltk.classify.accuracy(self.classifier, test_fs)
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

    def get_type(self):
        return ProblemTypeEnum.Classification

    def fill_with_predicted_data(self, ds, data):
        # TODO: Implement filling data (list of dictionaries)
        # with predicted values on ds dataset
        pass


class LinearRegression(AI):
    def __init__(self):
        self.regression = lm.LinearRegression()

    def run(self, ds_train, ds_test):
        X_train = ds_train['input']
        y_train = ds_train['class']
        X_test = ds_test['input']
        y_test = ds_test['class']
        self.regression.fit(X_train, y_train)
        tstresult = self.regression.score(X_test, y_test)
        return tstresult

    def test(self, ds_test):
        X_test = ds_test['input']
        y_test = ds_test['class']
        tstresult = self.regression.score(X_test, y_test)
        return tstresult

    def run_with_crossvalidation(self, ds, iterations = 5):
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            ds_train = ds[train_index, :]
            ds_test = ds[test_index, :]

            X_train = ds_train['input']
            y_train = ds_train['class']
            X_test = ds_test['input']
            y_test = ds_test['class']
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

    def get_type(self):
        return ProblemTypeEnum.Regression

    def fill_with_predicted_data(self, ds, data):
        # TODO: Implement filling data (list of dictionaries)
        # with predicted values on ds dataset
        pass