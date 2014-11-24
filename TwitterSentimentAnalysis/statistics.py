# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import enum
import inspect
from django_chartit_1_7 import DataPool, Chart


class StatisticEnum(enum.Enum):
    sample = 'sample'
    followers_vs_sent_retw = 'Followers vs sentiment/retweet count'
    predicted_retweets_vs_real_retweets = 'Predicted retweets vs real retweets'

    @classmethod
    def choices(cls):
        # get all members of the class
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        # filter down to just properties
        props = [m for m in members if not(m[0][:2] == '__')]
        # format into django choice tuple
        return tuple([(p[0], p[1].value) for p in props])


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
        if stat_type == StatisticEnum.followers_vs_sent_retw:
            return TweetStatistics.get_followers_vs_x(data, ai_model)



    @staticmethod
    def get_sample_chart(data, ai_model):
        if ai_model.problem_type == 1:  # classification
            terms = ['followers_count', 'sentiment_estimated', 'sentiment_actual']
            terms_dict = {'followers_count': ['sentiment_estimated', 'sentiment_actual']}
            title = 'Estimated sentiment vs followers count'
            x_axis = "Followers count"
            y_axis = "Sentiment"
        else:  # regression
            terms = ['followers_count', 'retweet_count_estimated', 'retweet_count_actual']
            terms_dict = {'followers_count': ['retweet_count_estimated', 'retweet_count_actual']}
            title = 'Estimated retweet count vs followers count'
            x_axis = "Followers count"
            y_axis = "Retweet count"

        # Create data structure for charts
        data = DataPool(
            series=[{
                'options': {
                    'source': data.order_by('followers_count')},
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

                'xAxis': {
                    'labels':{
                        'step' : '10',
                        'maxStaggerLines' : '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    @staticmethod
    def get_followers_vs_x(data, ai_model):
        if ai_model.problem_type == 1:  # classification
            terms = ['followers_count', 'sentiment_estimated', 'sentiment_actual']
            terms_dict = {'followers_count': ['sentiment_estimated', 'sentiment_actual']}
            title = 'Estimated sentiment vs followers count'
            x_axis = "Followers count"
            y_axis = "Sentiment"
        else:  # regression
            terms = ['followers_count', 'retweet_count_estimated', 'retweet_count_actual']
            terms_dict = {'followers_count': ['retweet_count_estimated', 'retweet_count_actual']}
            title = 'Estimated retweet count vs followers count'
            x_axis = "Followers count"
            y_axis = "Retweet count"

        # Create data structure for charts
        data = DataPool(
            series=[{
                'options': {
                    'source': data.order_by('followers_count')},
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

                'xAxis': {
                    'labels':{
                        'step' : '10',
                        'maxStaggerLines' : '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht
