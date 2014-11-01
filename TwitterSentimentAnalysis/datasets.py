# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import inject
from config import Config
from pymongo import MongoClient
from pybrain.datasets import ClassificationDataSet
from abc import ABCMeta, abstractmethod


class DatasetFactory(object):
    __metaclass__ = ABCMeta

    @abstractmethod
    def get_dataset(self):
        pass


class SimpleTweetDatasetFactory(DatasetFactory):
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
        return ClassificationDataSet(3, 1, 9, class_labels=SimpleTweetDatasetFactory.labels)

    @staticmethod
    def convert_to_ds(x, y):
        ds = SimpleTweetDatasetFactory.__create_classification_dataset()
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
        manual_grade = 0
        if record['manual_grade'] == 'positive':
            manual_grade = 1
        elif record['manual_grade'] == 'negative':
            manual_grade = -1

        return round(word_grade * manual_grade)

    def get_dataset(self):
        ds = self.__create_classification_dataset()
        for record in self.db.train_data.find({"isActive": True}):
            ds.addSample(self.__get_input_from_record(record), self.__get_output_from_record(record))
        return ds


