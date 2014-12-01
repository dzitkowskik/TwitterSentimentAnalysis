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
        """
        A method to return all possible choices for AI enum values

        :return: a list of tuples of all possible enum choices
        """
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        props = [m for m in members if not(m[0][:2] == '__')]
        return tuple([(p[0], p[1].value) for p in props])


class AI(object):
    """
    This is an abstract class used as a base for all artificial intelligence classes in this project.
    It contains abstract methods creating an interface for all AI. Also it contains reusable methods
    implementing a factory pattern for AI classes.
    """

    __metaclass__ = ABCMeta

    @staticmethod
    def factory(ai_type):
        """
        A factory method for artificial intelligence classes
        :param ai_type: Type as an object of AIEnum of AI to create
        :return: Created object of AI
        """
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
        """
        Converts input and target values to a featureset with target values
        :param inpt: input values for featureset
        :param target: target values
        :return: a features set
        """
        result = []
        for i, k in enumerate(inpt):
            features = {'first': k[0], 'second': k[1], 'third': k[2], 'fourth': k[3]}
            result.append((features, target[i][0]))
        return result

    @staticmethod
    def fill_data_regression(target, results, data):
        """
        Fills data dictionary with retweet count number and predicted retweet count number from target and results
        :param target: actual retweet counts
        :param results: predicted retweet counts
        :param data: dictionary to fill
        """
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
    def __init__(self, inp_cnt=4, out_cnt=9, hid_cnt=5, max_epochs=50, con_epochs=4):
        """
        This function builds an Artificial Neural Network with a specified number of hidden layers.
        Args:
          hidden_counts (int): The number of units in the hidden layer.

        Returns:
          self (MultiClassClassificationNeuralNetwork): the function returns
          an instance of its class, with the neural network initialized.
        """
        self.hid_cnt = hid_cnt
        self.out_cnt = out_cnt
        self.inp_cnt = inp_cnt
        self.max_epochs = max_epochs
        self.con_epochs = con_epochs
        self.network = self.__build_default_network()

    def apply_custom_network(self, hidden_counts):
        """
        Changes a network to a new one with possibly multiple layers with various hidden neurons count
        :param hidden_counts: an array of numbers of hidden nodes in every hidden layer. For example:
            [3, 4, 5] means a NN with 3 hidden layers with 3 hidden neurons on 1st layer and so on...
        :return: self
        """
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
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the neural network.
        """
        out = self.network.activateOnDataset(ds_test)
        result = np.ravel(np.argmax(out, 1))
        error = percentError(result, ds_test['class'])

        return error

    def __build_default_network(self):
        return buildNetwork(self.inp_cnt, self.hid_cnt, self.out_cnt, outclass=SoftmaxLayer, bias=True)

    # noinspection PyProtectedMember
    def run(self, ds_train, ds_test):
        """
        This function both trains the ANN and evaluates the ANN using a specified training and testing set
        Args:
          ds_train (TweetClassificationDatasetFactory): the training dataset the neural network is trained with.
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the neural network.
        """
        ds_train._convertToOneOfMany()
        ds_test._convertToOneOfMany()

        trainer = BackpropTrainer(
            self.network,
            dataset=ds_train,
            momentum=0.1,
            verbose=True,
            weightdecay=0.01)

        trainer.trainUntilConvergence(
            dataset=ds_train,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)
        result = trainer.testOnClassData(dataset=ds_test)
        error = percentError(result, ds_test['class'])

        return error

    # noinspection PyProtectedMember
    def run_with_crossvalidation(self, ds, iterations=5):
        """
        This function estimates the performance of the neural network using crossvalidation using a specified dataset.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to crossvalidate the neural network.
          iterations (int, optional): number of iterations for the crossvalidation.
        Returns:
          error (float): the average percent error of the dataset, tested on the neural network using crossvalidation.
        """
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

            trainer.trainUntilConvergence(
                dataset=ds_train,
                maxEpochs=self.max_epochs,
                continueEpochs=self.con_epochs)
            errors[i] = percentError(
                trainer.testOnClassData(dataset=ds_test),
                ds_test['class'])
            i += 1

        print "Multi class NN cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        """
        This function saves the neural network.
        Args:
          path (String): the path where the neural network is going to be saved.
        """
        NetworkWriter.writeToFile(self.network, path)

    def load(self, path):
        """
        This function loads the neural network.
        Args:
          path (String): the path where the neural network is going to be loaded from.
        """
        self.network = NetworkReader.readFrom(path)

    def get_type(self):
        """
        This function returns the type of problem and type of artificial intelligence.
        Returns:
          ProblemTypeEnum.Classification (enum): the type of problem.
          AIEnum.MultiClassClassificationNeuralNetwork (enum): the type of artificial intelligence.
        """
        return ProblemTypeEnum.Classification, AIEnum.MultiClassClassificationNeuralNetwork

    def fill_with_predicted_data(self, ds, data):
        """
        This function fills a dataset with the real and predicted values using a specified database.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to fill the
          dictionary with the real and predicted values.
          data (dictionary): dataset gets filled with the real and the predicted values.
        """
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
    def __init__(self, hid_cnt=10, max_epochs=50, con_epochs=4):
        self.hidden = hid_cnt
        self.network = None
        self.con_epochs = con_epochs
        self.max_epochs = max_epochs

    def run(self, ds_train, ds_test):
        """
        This function both trains the ANN and evaluates the ANN using a specified training and testing set
        Args:
          ds_train (TweetClassificationDatasetFactory): the training dataset the neural network is trained with.
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the neural network.
        """
        self.network = NNclassifier(ds_train)
        self.network.setupNN(hidden=self.hidden, verbose=True)
        self.network.Trainer.trainUntilConvergence(
            dataset=ds_train,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the neural network.
        """
        out = self.network.Trainer.module.activateOnDataset(ds_test)
        result = np.ravel(np.argmax(out, 1))
        error = percentError(result, ds_test['target'])
        return error

    # noinspection PyProtectedMember
    def run_with_crossvalidation(self, ds, iterations=5):
        """
        This function estimates the performance of the neural network using crossvalidation using a specified dataset.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to crossvalidate the neural network.
          iterations (int, optional): number of iterations for the crossvalidation.
        Returns:
          error (float): the average percent error of the dataset, tested on the neural network using crossvalidation.
        """
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
            self.network.Trainer.trainUntilConvergence(
                dataset=ds_train,
                maxEpochs=self.max_epochs,
                continueEpochs=self.con_epochs)
            tstresult = self.test(ds_test)
            errors[i] = tstresult[0]
            i += 1

        print "Simple Classification Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        """
        This function saves the neural network.
        Args:
          path (String): the path where the neural network is going to be saved.
        """
        file_object = open(path, 'w')
        pickle.dump(self.network, file_object)
        file_object.close()

    def load(self, path):
        """
        This function loads the neural network.
        Args:
          path (String): the path where the neural network is going to be loaded from.
        """
        file_object = open(path, 'r')
        self.network = pickle.load(file_object)

    def get_type(self):
        """
        This function returns the type of problem and type of artificial intelligence.
        Returns:
          ProblemTypeEnum.Classification (enum): the type of problem.
          AIEnum.SimpleClassificationNeuralNetwork (enum): the type of artificial intelligence.
        """
        return ProblemTypeEnum.Classification, AIEnum.SimpleClassificationNeuralNetwork

    def fill_with_predicted_data(self, ds, data):
        """
        This function fills a dataset with the real and predicted values using a specified database.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to
          fill the dictionary with the real and predicted values.
          data (dictionary): dataset gets filled with the real and the predicted values.
        """
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
        """
        This function evaluates the classifier using a test set and has the error-rate as an output.
        Args:
          ds_train (TweetClassificationDatasetFactory): the training dataset the classifier is trained with.
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the classifier.
        """
        x_train = ds_train['input']
        y_train = ds_train['target']
        train_fs = self.to_feature_set(x_train, y_train)
        self.classifier = nltk.NaiveBayesClassifier.train(train_fs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        """
        This function evaluates the classifier using a test set and has the error-rate as an output.
        Args:
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the classifier.
        """
        x_test = ds_test['input']
        y_test = ds_test['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify_many([rec[0] for rec in test_fs])
        error = percentError(result, y_test)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        """
        This function estimates the performance of the classifier using crossvalidation using a specified dataset.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to crossvalidate the classifier.
          iterations (int, optional): number of iterations for the crossvalidation.
        Returns:
          error (float): the average percent error of the dataset, tested on the classifier using crossvalidation.
        """
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
            result = self.classifier.classify_many([rec[0] for rec in test_fs])
            error = percentError(result, y_test)
            errors[q] = error
            q += 1

        print "Naive Bayes Classifier cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        """
        This function saves the classifier.
        Args:
          path (String): the path where the classifier is going to be saved.
        """
        f = open(path, 'wb')
        pickle.dump(self.classifier, f)
        f.close()

    def load(self, path):
        """
        This function loads the classifier.
        Args:
          path (String): the path where the classifier is going to be loaded from.
        """
        f = open(path)
        self.classifier = pickle.load(f)
        f.close()

    def get_type(self):
        """
        This function returns the type of problem and type of classifier.
        Returns:
          ProblemTypeEnum.Classification (enum): the type of problem.
          AIEnum.NaiveBayesClassifier (enum): the type of artificial intelligence.
        """
        return ProblemTypeEnum.Classification, AIEnum.NaiveBayesClassifier

    def fill_with_predicted_data(self, ds, data):
        """
        This function fills a dataset with the real and predicted values using a specified database.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to
          fill the dictionary with the real and predicted values.
          data (dictionary): dataset gets filled with the real and the predicted values.
        """
        x_test = ds['input']
        y_test = ds['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify_many([rec[0] for rec in test_fs])
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
        """
        This function evaluates the classifier using a test set and has the error-rate as an output.
        Args:
          ds_train (TweetClassificationDatasetFactory): the training dataset the classifier is trained with.
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the classifier.
        """
        x_train = ds_train['input']
        y_train = ds_train['target']
        train_fs = self.to_feature_set(x_train, y_train)
        self.classifier = nltk.MaxentClassifier.train(train_fs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        """
        This function evaluates the classifier using a test set and has the error-rate as an output.
        Args:
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the classifier.
        """
        x_test = ds_test['input']
        y_test = ds_test['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify_many([rec[0] for rec in test_fs])
        error = percentError(result, y_test)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        """
        This function estimates the performance of the classifier using crossvalidation using a specified dataset.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset used to crossvalidate the classifier.
          iterations (int, optional): number of iterations for the crossvalidation.
        Returns:
          error (float): the average percent error of the dataset, tested on the classifier using crossvalidation.
        """
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
            result = self.classifier.classify_many([rec[0] for rec in test_fs])
            error = percentError(result, y_test)
            errors[q] = error
            q += 1

        print "Max Entropy Classifier cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        """
        This function saves the classifier.
        Args:
          path (String): the path where the classifier is going to be saved.
        """
        file_object = open(path, 'w')
        pickle.dump(self.classifier, file_object)
        file_object.close()

    def load(self, path):
        """
        This function loads the classifier.
        Args:
          path (String): the path where the classifier is going to be loaded from.
        """
        file_object = open(path, 'r')
        self.classifier = pickle.load(file_object)

    def get_type(self):
        """
        This function returns the type of problem and type of classifier.
        Returns:
          ProblemTypeEnum.Classification (enum): the type of problem.
          AIEnum.MaxEntropyClassifier (enum): the type of artificial intelligence.
        """
        return ProblemTypeEnum.Classification, AIEnum.MaxEntropyClassifier

    def fill_with_predicted_data(self, ds, data):
        """
        This function fills a dataset with the real and predicted values using a specified database.
        Args:
          ds (TweetClassificationDatasetFactory): the dataset
          used to fill the dictionary with the real and predicted values.
          data (dictionary): dataset gets filled with the real and the predicted values.
        """
        x_test = ds['input']
        y_test = ds['target']
        test_fs = self.to_feature_set(x_test, y_test)
        result = self.classifier.classify_many([rec[0] for rec in test_fs])
        middle = len(TweetClassificationDatasetFactory.labels) / 2
        i = 0
        assert(len(ds) == len(data))
        for record in data:
            record['sentiment'] = ds['target'][i] - middle
            record['predicted_sentiment'] = result[i] - middle
            i += 1


##############
# REGRESSION #
##############


class SimpleRegressionNeuralNetwork(AI):
    def __init__(self, hid_cnt=10, max_epochs=50, con_epochs=4):
        self.hidden = hid_cnt
        self.network = None
        self.con_epochs = con_epochs
        self.max_epochs = max_epochs

    def run(self, ds_train, ds_test):
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
          ds_train (TweetClassificationDatasetFactory): the training dataset the neural network is trained with.
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the neural network.
        """
        self.network = NNregression(ds_train)
        self.network.setupNN(hidden=self.hidden, verbose=True)
        self.network.Trainer.trainUntilConvergence(
            dataset=ds_train,
            maxEpochs=self.max_epochs,
            continueEpochs=self.con_epochs)
        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        """
        This function evaluates the ANN using a test set and has the error-rate as an output.
        Args:
          ds_test (TweetRegressionDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the network.
        """
        result = self.network.Trainer.module.activateOnDataset(ds_test)
        error = mean_squared_error(ds_test['target'], result)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        """
        This function estimates the performance of the neural network using crossvalidation using a specified dataset.
        Args:
          ds (TweetRegressionDatasetFactory): the dataset used to crossvalidate the network.
          iterations (int, optional): number of iterations for the crossvalidation.
        Returns:
          error (float): the average percent error of the dataset, tested on the network using crossvalidation.
        """
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
            self.network.Trainer.trainUntilConvergence(
                dataset=ds_train,
                maxEpochs=self.max_epochs,
                continueEpochs=self.con_epochs)
            tstresult = self.test(ds_test)
            errors[i] = tstresult[0]
            i += 1

        print "Simple Regression Neural Network cross-validation test errors: " % errors
        return np.average(errors)

    def __call__(self, ds_train, ds_test):
        return self.run(ds_train, ds_test)

    def save(self, path):
        """
        This function saves the neural network.
        Args:
          path (String): the path where the network is going to be saved.
        """
        file_object = open(path, 'w')
        pickle.dump(self.network, file_object)
        file_object.close()

    def load(self, path):
        """
        This function loads the neural network.
        Args:
          path (String): the path where the neural network is going to be loaded from.
        """
        file_object = open(path, 'r')
        self.network = pickle.load(file_object)

    def get_type(self):
        """
        This function returns the type of problem and type of neural network.
        Returns:
          ProblemTypeEnum.Regression (enum): the type of problem.
          AIEnum.SimpleRegressionNeuralNetwork (enum): the type of artificial intelligence.
        """
        return ProblemTypeEnum.Regression, AIEnum.SimpleRegressionNeuralNetwork

    def fill_with_predicted_data(self, ds, data):
        """
        This function fills a dataset with the real and predicted values using a specified database.
        Args:
          ds (TweetRegressionDatasetFactory): the dataset used to fill the dictionary with the real and predicted values.
          data (dictionary): dataset gets filled with the real and the predicted values.
        """
        target = ds['target']
        out = self.network.Trainer.module.activateOnDataset(ds)
        results = np.ravel(np.argmax(out, 1))
        self.fill_data_regression(target, results, data)


class LinearRegression(AI):
    def __init__(self):
        self.regression = lm.LinearRegression()

    def run(self, ds_train, ds_test):
        """
        This function evaluates the model using a test set and has the error-rate as an output.
        Args:
          ds_train (TweetClassificationDatasetFactory): the training dataset the model is trained with.
          ds_test (TweetClassificationDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the model.
        """
        x_train = ds_train['input']
        y_train = ds_train['target']
        self.regression.fit(x_train, y_train)

        error = self.test(ds_test)
        return error

    def test(self, ds_test):
        """
        This function evaluates the model using a test set and has the error-rate as an output.
        Args:
          ds_test (TweetRegressionDatasetFactory): the test dataset evaluated.
        Returns:
          error (float): the percent error of the test dataset, tested on the model.
        """
        result = self.regression.predict(ds_test['input'])
        error = mean_squared_error(ds_test['target'], result)
        return error

    def run_with_crossvalidation(self, ds, iterations=5):
        """
        This function estimates the performance of the model using crossvalidation using a specified dataset.
        Args:
          ds (TweetRegressionDatasetFactory): the dataset used to crossvalidate the model.
          iterations (int, optional): number of iterations for the crossvalidation.
        Returns:
          error (float): the average percent error of the dataset, tested on the classifier using crossvalidation.
        """
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
        """
        This function saves the model.
        Args:
          path (String): the path where the network is going to be saved.
        """
        f = open(path, 'wb')
        pickle.dump(self.regression, f)
        f.close()

    def load(self, path):
        """
        This function loads the model.
        Args:
          path (String): the path where the model is going to be loaded from.
        """
        f = open(path)
        self.regression = pickle.load(f)
        f.close()

    def get_type(self):
        """
        This function returns the type of problem and type of model.
        Returns:
          ProblemTypeEnum.Regression (enum): the type of problem.
          AIEnum.LinearRegression (enum): the type of artificial intelligence.
        """
        return ProblemTypeEnum.Regression, AIEnum.LinearRegression

    def fill_with_predicted_data(self, ds, data):
        """
        This function fills a dataset with the real and predicted values using a specified database.
        Args:
          ds (TweetRegressionDatasetFactory): the dataset used to fill the dictionary with the real and predicted values.
          data (dictionary): dataset gets filled with the real and the predicted values.
        """
        target = ds['target']
        results = self.regression.predict(ds['input'])
        self.fill_data_regression(target, results, data)