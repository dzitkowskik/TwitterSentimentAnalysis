# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import enum
from django_chartit_1_7 import DataPool, Chart


class StatisticEnum(enum.Enum):
    sample = 'sample'


# TODO: Implement more statistics similar to sample one
# 1) ai_model is an object of ArtificialIntelligence class from models
# 2) get_chart for any stat_type should return an object of Chart class
# from chartit
# 3) We need multiple chart types using various columns of data
# 4) Charts are base on Highcharts so you can find all information
# how to create and modify charts in their web page:
# http://www.highcharts.com/docs/getting-started/your-first-chart
# be careful ! - YOU MUST USE PYTHON SYNTAX FOR THAT !
# 5) Documentation for django chartit (library we use for charts) is
# available on: http://chartit.shutupandship.com/docs/ (really poor)
class TweetStatistics(object):
    @staticmethod
    def get_chart(stat_type, data, ai_model):
        if stat_type == StatisticEnum.sample:
            return TweetStatistics.get_sample_chart(data, ai_model)

    @staticmethod
    def get_sample_chart(data, ai_model):
        if ai_model.problem_type == 1:  # classification
            terms = ['followers_count', 'sentiment_estimated']
            terms_dict = {'sentiment_estimated': ['followers_count']}
            title = 'Estimated sentiment vs followers count'
        else:  # regression
            terms = ['followers_count', 'retweet_count_estimated']
            terms_dict = {'retweet_count_estimated': ['followers_count']}
            title = 'Estimated retweet count vs followers count'

        # Create data structure for charts
        data = DataPool(
            series=[{
                'options': {
                    'source': data},
                'terms': terms}])

        # Create chart
        cht = Chart(
            datasource=data,
            series_options=[{
                'options': {
                    'type': 'line',
                    'stacking': True},
                'terms': terms_dict}],
            chart_options={
                'title': {
                    'text': title},
                'yAxis': {
                    'title': {
                        'text': 'Followers count'}}})

        return cht
