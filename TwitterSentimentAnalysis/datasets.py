# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import inject
from config import Config
from pymongo import MongoClient
from pybrain.datasets import ClassificationDataSet
from pybrain.datasets import SupervisedDataSet
from abc import ABCMeta, abstractmethod
from datetime import datetime
import enum
import math


class ProblemTypeEnum(enum.Enum):
    Classification = 1
    Regression = 2


class DatasetFactory(object):
    """
    This is an abstract class used as a base for all dataset factories classes in this project.
    It contains abstract methods creating an interface for all dataset factories. This class
    also provides a suitable interface for a dataset factory.
    """
    __metaclass__ = ABCMeta

    @staticmethod
    def factory(problem_type):
        """
        This method creates a proper dataset factory for the problem type.
        :param problem_type: type of a problem of class ProblemTypeEnum (regression or classification)
        :return: a proper dataset factory object
        """
        if problem_type == ProblemTypeEnum.Classification:
            return TweetClassificationDatasetFactory()
        elif problem_type == ProblemTypeEnum.Regression:
            return TweetRegressionDatasetFactory()
        assert 0, "Bad enum given: " + str(problem_type)

    @abstractmethod
    def get_dataset(self, table_name, search_params):
        pass

    @abstractmethod
    def get_dataset_class(self, table_name, search_params):
        pass

    @abstractmethod
    def get_data(self, table_name, search_params):
        pass


class TweetClassificationDatasetFactory(DatasetFactory):
    """
    This class implements dataset factory interface, and produces data in suitable format for classification problems.
    """
    labels = [
        'Highly negative (-4)',
        'Fairly negative (-3)',
        'Moderately negative (-2)',
        'Lightly negative (-1)',
        'Neutral',
        'Lightly positive (+1)',
        'Moderately positive (+2)',
        'Fairly positive (+3)',
        'Highly positive (+4)']

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    @staticmethod
    def __create_classification_dataset():
        return ClassificationDataSet(4, 1, 9, class_labels=TweetClassificationDatasetFactory.labels)

    @staticmethod
    def convert_to_ds(x, y):
        """
        This method converts a matrices x and y of input and target values to a classification dataset
        :param x: input values as a matrix NxM where N is a number of records and M a number of features
        :param y: a target values corresponding to true labels of the records
        :return: an object of classification dataset containing passed data
        """
        ds = TweetClassificationDatasetFactory.__create_classification_dataset()
        for i in range(0, len(y)):
            ds.addSample(x[i], y[i])
        return ds

    @staticmethod
    def __get_input_from_record(record):
        favorite_count = record['data']['favorite_count']
        followers_count = record['data']['user']['followers_count']
        retweet_count = record['data']['retweet_count']
        age_of_tweet = (datetime.now() - datetime.strptime(
            record['data']['created_at'],
            '%a %b %d %H:%M:%S +0000 %Y'))

        return favorite_count, followers_count, retweet_count, age_of_tweet.seconds

    @staticmethod
    def __get_output_from_record(record):
        # word_sentiment is stored as float between -5 and 5
        scale = 4.0 / 5.0
        word_grade = abs(record['word_sentiment']) * scale  # between 0 and 4

        if record['manual_grade'] == 'positive':
            manual_grade = 1.0
        elif record['manual_grade'] == 'negative':
            manual_grade = -1.0
        else:
            manual_grade = math.copysign(1.0, record['word_sentiment'])

        # sentiment is 0-3 for negative where 0 is highly negative
        # 4 is neutral and values 5-8 are positive where 8 is highly positive
        return round(word_grade * manual_grade) + 4.0

    def get_dataset(self, table_name='train_data', search_params=None):
        """
        This method created a classification dataset from data queried from a database from specified table
        with specified search params as a dictionary
        :param table_name: a name of a table to fetch data from
        :param search_params: a dictionary of filters like: {"isActive": True}
        :return: a classification dataset containing data from db
        """
        search_params = search_params or {"isActive": True}
        ds = self.__create_classification_dataset()
        data = self.get_data(table_name, search_params)
        for record in data:
            inp = self.__get_input_from_record(record)
            target = self.__get_output_from_record(record)
            ds.addSample(inp, target)
        return ds

    def get_dataset_class(self, table_name='train_data', search_params=None):
        """
        The same as get_dataset method but returns a featureset
        """
        search_params = search_params or {"isActive": True}
        data = self.db[table_name].find(search_params)
        featureset = [(self.__get_input_from_record(record), self.__get_output_from_record(record)) for record in data]
        return featureset

    def get_data(self, table_name='train_data', search_params=None):
        """
        The same as get_dataset method but returns only fetched data from db as pymongo Cursor.
        """
        search_params = search_params or {"isActive": True}
        return self.db[table_name].find(search_params)


class TweetRegressionDatasetFactory(DatasetFactory):
    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    # input :   - sentiment of tweet
    #           - number of followers
    #           - age of tweet
    #           - number of favorites
    @staticmethod
    def __create_regression_dataset():
        return SupervisedDataSet(4, 1)

    @staticmethod
    def convert_to_ds(x, y):
        """
        This method converts a matrices x and y of input and target values to a regression dataset
        :param x: input values as a matrix NxM where N is a number of records and M a number of features
        :param y: a target values corresponding to true labels of the records
        :return: an object of regression dataset containing passed data
        """
        ds = TweetRegressionDatasetFactory.__create_regression_dataset()
        for i in range(0, len(y)):
            ds.addSample(x[i], y[i])
        return ds

    @staticmethod
    def __get_input_from_record(record):
        favorite_count = record['data']['favorite_count']
        followers_count = record['data']['user']['followers_count']
        word_sentiment = record['word_sentiment']
        age_of_tweet = (datetime.now() - datetime.strptime(
            record['data']['created_at'],
            '%a %b %d %H:%M:%S +0000 %Y'))

        return favorite_count, followers_count, word_sentiment, age_of_tweet.seconds

    @staticmethod
    def __get_output_from_record(record):
        retweet_count = record['data']['retweet_count']
        return retweet_count

    def get_dataset(self, table_name='train_data', search_params=None):
        """
        This method created a regression dataset from data queried from a database from specified table
        with specified search params as a dictionary
        :param table_name: a name of a table to fetch data from
        :param search_params: a dictionary of filters like: {"isActive": True}
        :return: a regression dataset containing data from db
        """
        search_params = search_params or {"isActive": True}
        ds = self.__create_regression_dataset()
        data = self.get_data(table_name, search_params)
        for record in data:
            ds.addSample(self.__get_input_from_record(record), self.__get_output_from_record(record))
        return ds

    def get_dataset_class(self, table_name='train_data', search_params=None):
        """
        The same as get_dataset method but returns a featureset
        """
        search_params = search_params or {"isActive": True}
        data = self.db[table_name].find(search_params)
        featureset = [(self.__get_input_from_record(record), self.__get_output_from_record(record)) for record in data]
        return featureset

    def get_data(self, table_name='train_data', search_params=None):
        """
        The same as get_dataset method but returns only fetched data from db as pymongo Cursor.
        """
        search_params = search_params or {"isActive": True}
        return self.db[table_name].find(search_params)