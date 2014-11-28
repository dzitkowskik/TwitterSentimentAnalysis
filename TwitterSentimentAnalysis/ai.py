# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import enum
import nltk
import pickle
import inspect
import numpy as np
import sklearn.linear_model as lm
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.shortcuts import buildNetwork
from pybrain import FeedForwardNetwork, LinearLayer, SigmoidLayer, FullConnection, SoftmaxLayer
from abc import ABCMeta, abstractmethod
from sklearn import cross_validation
from TwitterSentimentAnalysis.datasets import TweetClassificationDatasetFactory, ProblemTypeEnum
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader
from pybrain.tools.neuralnets import NNregression
from pybrain.tools.neuralnets import NNclassifier
from sklearn.metrics import mean_squared_error


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
            nn = MultiClassClassificationNeuralNetwork()
            nn.apply_custom_network([4, 9])
            return nn
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
    def to_feature_set(inpt, target):
        result = []
        for i, k in enumerate(inpt):
            features = {'first': k[0], 'second': k[1], 'third': k[2], 'fourth': k[3]}
            result.append((features, target[i][0]))
        return result

    @staticmethod
    def fill_data_regression(target, results, data):
        i = 0
        assert(len(results) == len(data))
        for record in data:
            rc = round(target[i])
            prc = round(results[i])
            record['retweet_count'] = rc if rc > 0 else 0
            record['predicted_retweet_count'] = prc if prc > 0 else 0
            i += 1

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


##################
# CLASSIFICATION #
##################


class MultiClassClassificationNeuralNetwork(AI):
    def __init__(self, inp_cnt=4, out_cnt=9, hid_cnt=5, epochs=20):
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
        out = self.network.activateOnDataset(ds_test)
        result = np.ravel(np.argmax(out, 1))
        error = percentError(result, ds_test['class'])

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
        result = trainer.testOnClassData(dataset=ds_test)
        error = percentError(result, ds_test['class'])

        return error

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
        return ProblemTypeEnum.Classification, AIEnum.MultiClassClassificationNeuralNetwork

    def fill_with_predicted_data(self, ds, data):
        out = self.network.activateOnDataset(ds)
        results = np.ravel(np.argmax(out, 1))
        middle = len(TweetClassificationDatasetFactory.labels) / 2
        i = 0
        assert(len(ds) == len(data))
        for record in data:
            record['sentiment'] = ds['target'][i] - middle
            record['predicted_sentiment'] = results[i] - middle
            i += 1


class SimpleClassificationNeuralNetwork(AI):
    def __init__(self, hid_cnt=10, max_epochs=100, con_epochs=4):
        self.hidden = hid_cnt
        self.network = None
        self.con_epochs = con_epochs
        self.max_epochs = max_epochs

    def run(self, ds_train, ds_test):
        self.network = NNclassifier(ds_train)
        self.network.setupNN(hidden=self.hidden, verbose=True)
        self.network.Trainer.trainUntilConvergence(
            dataset=ds_train,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        out = self.network.Trainer.module.activateOnDataset(ds_test)
        result = np.ravel(np.argmax(out, 1))
        error = percentError(result, ds_test['target'])
        return error

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
            self.network.setupNN(hidden=self.hidden)
            self.network.runTraining(self.convergence)
            tstresult = self.test(ds_test)
            errors[i] = tstresult[0]
            i += 1

        print "Simple Classification Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        file_object = open(path, 'w')
        pickle.dump(self.network, file_object)
        file_object.close()

    def load(self, path):
        file_object = open(path, 'r')
        self.network = pickle.load(file_object)

    def get_type(self):
        return ProblemTypeEnum.Classification, AIEnum.SimpleClassificationNeuralNetwork

    def fill_with_predicted_data(self, ds, data):
        out = self.network.Trainer.module.activateOnDataset(ds)
        results = np.ravel(np.argmax(out, 1))
        middle = len(TweetClassificationDatasetFactory.labels) / 2
        i = 0
        assert(len(ds) == len(data))
        for record in data:
            record['sentiment'] = ds['target'][i] - middle
            record['predicted_sentiment'] = results[i] - middle
            i += 1


class NaiveBayesClassifier(AI):
    def __init__(self):
        self.classifier = None

    def run(self, ds_train, ds_test):
        x_train = ds_train['input']
        y_train = ds_train['target']
        train_fs = self.to_feature_set(x_train, y_train)
        self.classifier = nltk.NaiveBayesClassifier.train(train_fs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        x_test = ds_test['input']
        y_test = ds_test['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify(test_fs)
        error = percentError(result, y_test)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        x = ds['input']
        y = ds['target']
        n, m = x.shape
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        q = 0
        for train_index, test_index in cv:
            x_train = x[train_index, :]
            y_train = y[train_index, :]
            x_test = x[test_index, :]
            y_test = y[test_index, :]

            train_fs = self.to_feature_set(x_train, y_train)
            self.classifier = nltk.NaiveBayesClassifier.train(train_fs)

            test_fs = self.to_feature_set(x_test, y_test)
            result = self.classifier.classify(test_fs)
            error = percentError(result, y_test)
            errors[q] = error
            q += 1

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
        return ProblemTypeEnum.Classification, AIEnum.NaiveBayesClassifier

    def fill_with_predicted_data(self, ds, data):
        x_test = ds['input']
        y_test = ds['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify(test_fs)
        middle = len(TweetClassificationDatasetFactory.labels) / 2
        i = 0
        assert(len(ds) == len(data))
        for record in data:
            record['sentiment'] = ds['target'][i] - middle
            record['predicted_sentiment'] = result[i] - middle
            i += 1


class MaxEntropyClassifier(AI):
    def __init__(self):
        self.classifier = None

    def run(self, ds_train, ds_test):
        x_train = ds_train['input']
        y_train = ds_train['target']
        train_fs = self.to_feature_set(x_train, y_train)
        self.classifier = nltk.MaxentClassifier.train(train_fs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        x_test = ds_test['input']
        y_test = ds_test['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify(test_fs)
        error = percentError(result, y_test)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        x = ds['input']
        y = ds['target']
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        q = 0
        for train_index, test_index in cv:
            x_train = x[train_index, :]
            y_train = y[train_index, :]
            x_test = x[test_index, :]
            y_test = y[test_index, :]

            train_fs = self.to_feature_set(x_train, y_train)
            self.classifier = nltk.MaxentClassifier.train(train_fs)

            test_fs = self.to_feature_set(x_test, y_test)
            result = self.classifier.classify(test_fs)
            error = percentError(result, y_test)
            errors[q] = error
            q += 1

        print "Max Entropy Classifier cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        file_object = open(path, 'w')
        pickle.dump(self.classifier, file_object)
        file_object.close()

    def load(self, path):
        file_object = open(path, 'r')
        self.classifier = pickle.load(file_object)

    def get_type(self):
        return ProblemTypeEnum.Classification, AIEnum.MaxEntropyClassifier

    def fill_with_predicted_data(self, ds, data):
        x_test = ds['input']
        y_test = ds['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify(test_fs)
        i = 0
        assert(len(ds) == len(data))
        for record in data:
            record['sentiment'] = ds['target'][i]
            record['predicted_sentiment'] = result[i]
            i += 1


##############
# REGRESSION #
##############


class SimpleRegressionNeuralNetwork(AI):
    def __init__(self, hid_cnt=10, max_epochs=100, con_epochs=4):
        self.hidden = hid_cnt
        self.network = None
        self.con_epochs = con_epochs
        self.max_epochs = max_epochs

    def run(self, ds_train, ds_test):
        self.network = NNregression(ds_train)
        self.network.setupNN(hidden=self.hidden, verbose=True)
        self.network.Trainer.trainUntilConvergence(
            dataset=ds_train,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        result = self.network.Trainer.module.activateOnDataset(ds_test)
        error = mean_squared_error(ds_test['target'], result)
        return error

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
            self.network.setupNN(hidden=self.hidden)
            self.network.runTraining(self.convergence)
            tstresult = self.test(ds_test)
            errors[i] = tstresult[0]
            i += 1

        print "Simple Regression Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        file_object = open(path, 'w')

        pickle.dump(self.network, file_object)

        file_object.close()

    def load(self, path):
        file_object = open(path, 'r')
        self.network = pickle.load(file_object)

    def get_type(self):
        return ProblemTypeEnum.Regression, AIEnum.SimpleRegressionNeuralNetwork

    def fill_with_predicted_data(self, ds, data):
        target = ds['target']
        out = self.network.Trainer.module.activateOnDataset(ds)
        results = np.ravel(np.argmax(out, 1))
        self.fill_data_regression(target, results, data)


class LinearRegression(AI):
    def __init__(self):
        self.regression = lm.LinearRegression()

    def run(self, ds_train, ds_test):
        x_train = ds_train['input']
        y_train = ds_train['target']
        self.regression.fit(x_train, y_train)

        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        result = self.regression.predict(ds_test['input'])
        error = mean_squared_error(ds_test['target'], result)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        x = ds['input']
        y = ds['target']
        n = len(ds)
        cv = cross_validation.KFold(n, iterations, shuffle=True)
        errors = np.zeros(iterations)

        i = 0
        for train_index, test_index in cv:
            x_train = x[train_index, :]
            y_train = y[train_index, :]
            x_test = x[test_index, :]
            y_test = y[test_index, :]

            self.regression.fit(x_train, y_train)
            errors[i] = self.regression.score(x_test, y_test)
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
        return ProblemTypeEnum.Regression, AIEnum.LinearRegression

    def fill_with_predicted_data(self, ds, data):
        target = ds['target']
        results = self.regression.predict(ds['input'])
        self.fill_data_regression(target, results, data)