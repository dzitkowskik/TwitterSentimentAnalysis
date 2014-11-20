# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import inject
from config import Config
from pymongo import MongoClient
from django_chartit_1_7 import DataPool, Chart


class TweetStatistics(object):
    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]


    @staticmethod
    def get_sample_chart(data, statistic=None):

        data = DataPool(
            series=[{
                'options': {
                    'source': data},
                'terms': ['retweet_count_actual', 'followers_count']}])

        cht = Chart(
            datasource=data,
            series_options=[{
                'options': {
                    'type': 'column',
                    'stacking': True},
                'terms': {
                    'followers_count': [
                        'retweet_count_actual']}}],
            chart_options={
                'title': {
                    'text': 'Retweet count vs Sentiment'},
                'xAxis': {
                    'title': {
                        'text': 'Retweet count'}}})

        return cht
