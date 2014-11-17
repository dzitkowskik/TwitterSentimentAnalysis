# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import inject
from config import Config
from pymongo import MongoClient
from chartit import DataPool
from TwitterSentimentAnalysis.datasets import DatasetFactory, ProblemTypeEnum


class TweetStatistics(object):
    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    @staticmethod
    def get_sample_stat():
        factory = DatasetFactory.factory(ProblemTypeEnum.Classification)
        source = factory.get_data()
        data = DataPool(
            series=[{
                'options': {
                    'source': source},
                'terms': ['retweet_count', 'sentiment']}])
        return data