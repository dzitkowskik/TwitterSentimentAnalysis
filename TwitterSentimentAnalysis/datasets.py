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
import datetime
import enum
import math


class ProblemTypeEnum(enum.Enum):
    Classification = 1
    Regression = 2


class DatasetFactory(object):
    __metaclass__ = ABCMeta

    @staticmethod
    def factory(problem_type):
        if problem_type == ProblemTypeEnum.Classification:
            return TweetClassificationDatasetFactory()
        elif problem_type == ProblemTypeEnum.Regression:
            return TweetRegressionDatasetFactory()
        assert 0, "Bad enum given: " + str(problem_type)

    @abstractmethod
    def get_dataset(self):
        pass


class TweetClassificationDatasetFactory(DatasetFactory):
    labels = [
        'Negative (-4)',
        'Negative (-3)',
        'Negative (-2)',
        'Negative (-1)',
        'Neutral',
        'Positive (+1)',
        'Positive (+2)',
        'Positive (+3)',
        'Positive (+4)']

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    @staticmethod
    def __create_classification_dataset():
        return ClassificationDataSet(3, 1, 9, class_labels=TweetClassificationDatasetFactory.labels)

    @staticmethod
    def convert_to_ds(x, y):
        ds = TweetClassificationDatasetFactory.__create_classification_dataset()
        for i in range(0, len(y)):
            ds.addSample(x[i], y[i])
        return ds

    @staticmethod
    def __get_input_from_record(record):
        favorite_count = record['data']['favorite_count']
        followers_count = record['data']['user']['followers_count']
        retweet_count = record['data']['retweet_count']
        return favorite_count, followers_count, retweet_count

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

        return round(word_grade * manual_grade)

    def get_dataset(self, table_name='train_data', search_params={"isActive": True}):
        ds = self.__create_classification_dataset()
        data = self.db[table_name].find(search_params)
        for record in data:
            ds.addSample(self.__get_input_from_record(record), self.__get_output_from_record(record))
        return ds

    def get_dataset_class(self, table_name='train_data', search_params={"isActive": True}):
        data = self.db[table_name].find(search_params)
        featureset = [(self.__get_input_from_record(record), self.__get_output_from_record(record)) for record in data]
        return featureset


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
        ds = TweetRegressionDatasetFactory.__create_regression_dataset()
        for i in range(0, len(y)):
            ds.addSample(x[i], y[i])
        return ds

    @staticmethod
    def __get_input_from_record(record):
        favorite_count = record['data']['favorite_count']
        followers_count = record['data']['user']['followers_count']
        word_sentiment = record['sentiment']
        age_of_tweet = (datetime.datetime.now() - datetime.strptime(record['user']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')).days
        return favorite_count, followers_count, word_sentiment, age_of_tweet

    @staticmethod
    def __get_output_from_record(record):
        retweet_count = record['data']['retweet_count']
        return retweet_count

    def get_dataset(self, table_name='train_data', search_params={"isActive": True}):
        ds = self.__create_regression_dataset()
        data = self.db[table_name].find(search_params)
        for record in data:
            ds.addSample(self.__get_input_from_record(record), self.__get_output_from_record(record))
        return ds

    def get_dataset_class(self, table_name='train_data', search_params={"isActive": True}):
        data = self.db[table_name].find(search_params)
        featureset = [(self.__get_input_from_record(record), self.__get_output_from_record(record)) for record in data]
        return featureset