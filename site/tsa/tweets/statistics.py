# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import enum
import inspect
from django_chartit_1_7 import DataPool, Chart
from models import *
from django.db.models import Avg
import datetime


class StatisticEnum(enum.Enum):
    sample = 'sample'
    followers_vs_x = 'Followers vs sentiment (classifier)/retweet count (regression)'
    favorites_vs_x = 'Number of favorites vs sentiment (classifier)/retweet count (regression)'
    hourly = 'Hour of the day vs sentiment (classifier)/retweet count (regression)'
    day_of_week = 'Day of the week vs sentiment (classifier)/retweet count regression)'

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
        if stat_type == StatisticEnum.followers_vs_x:
            return TweetStatistics.get_followers_vs_x(data, ai_model)
        if stat_type == StatisticEnum.favorites_vs_x:
            return TweetStatistics.get_favorites_vs_x(data, ai_model)
        if stat_type == StatisticEnum.hourly:
            return TweetStatistics.get_hourly(ai_model)
        if stat_type == StatisticEnum.day_of_week:
            return TweetStatistics.get_day_of_week(ai_model)

    '''
    This function saves the data from the database to a model used to make the graphs.
    Args:
      data: the dataset used to fill the model with data.
      ai_name (model): the saved artificial intelligence.
    '''

    @staticmethod
    def save_data_for_statistics(data, ai_name):
        if ai_name != "" and ai_name is not None:
            ai = ArtificialIntelligence.objects.get(name=ai_name)
            for record in data:
                sentiment_estimated = record['predicted_sentiment'] \
                    if 'predicted_sentiment' in record else None
                retweet_count_estimated = record['predicted_retweet_count'] \
                    if 'predicted_retweet_count' in record else None
                sentiment = record['sentiment'] if 'sentiment' in record \
                    else record['word_sentiment']
                date = datetime.datetime.strptime(record['data']['created_at'], '%a %b %d %H:%M:%S +0000 %Y')
                new_tweet_save = Tweet(
                    number=record['_id'],
                    text=record['text'],
                    date=date,
                    hour=date.hour,
                    day_of_week=date.weekday(),
                    favourites_count=record['data']['favorite_count'],
                    followers_count=record['data']['user']['followers_count'],
                    retweet_count_actual=record['retweet_count'],
                    retweet_count_estimated=retweet_count_estimated,
                    sentiment_actual=sentiment,
                    sentiment_estimated=sentiment_estimated,
                    ai=ai
                )
                new_tweet_save.save()

            hour_vs_retweet_count = Tweet.objects.filter(ai=ai).values('hour')\
                .annotate(Avg('retweet_count_actual'), Avg('retweet_count_estimated'))
            for record in hour_vs_retweet_count:
                HourRetweet(
                    hour=record['hour'],
                    retweet_actual_avg=record['retweet_count_actual__avg'],
                    retweet_predicted_avg=record['retweet_count_estimated__avg'],
                    ai=ai
                ).save()

            hour_vs_sentiment = Tweet.objects.filter(ai=ai).values('hour')\
                .annotate(Avg('sentiment_actual'), Avg('sentiment_estimated'))
            for record in hour_vs_sentiment:
                HourSentiment(
                    hour=record['hour'],
                    sentiment_actual_avg=record['sentiment_actual__avg'],
                    sentiment_predicted_avg=record['sentiment_estimated__avg'],
                    ai=ai
                ).save()

            dayofweek_retweet_count = Tweet.objects.filter(ai=ai).values('day_of_week')\
                .annotate(Avg('retweet_count_actual'), Avg('retweet_count_estimated'))
            for record in dayofweek_retweet_count:
                DayofweekRetweet(
                    retweet_actual_avg=record['retweet_count_actual__avg'],
                    retweet_predicted_avg=record['retweet_count_estimated__avg'],
                    day_of_week=record['day_of_week'],
                    ai=ai
                ).save()

            dayofweek_vs_sentiment = Tweet.objects.filter(ai=ai).values('day_of_week')\
                .annotate(Avg('sentiment_actual'), Avg('sentiment_estimated'))
            for record in dayofweek_vs_sentiment:
                DayofweekSentiment(
                    sentiment_actual_avg=record['sentiment_actual__avg'],
                    sentiment_predicted_avg=record['sentiment_estimated__avg'],
                    day_of_week=record['day_of_week'],
                    ai=ai
                ).save()

        else:
            raise NameError('ai_name cannot be blank')

    '''
    This function returns a sample chart.
    Args:
      data: the data used to build the chart.
      ai_name (model): the problem type.
    Returns:
      cht (Chart): the resulting graph to be shown on the webpage.
    '''

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

        data.values('hour').annotate(Avg('retweet_count_actual'), Avg('retweet_count_estimated'))
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
                    'labels': {
                        'step': '10',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    '''
    This function returns the day of the week versus predicted and actual sentiment (classification)
    or the day of the week versus actual retweet count and predicted retweet count (regression).
    Args:
      data: the data used to build the chart.
      ai_name (model): the problem type.
    Returns:
      cht (Chart): the resulting graph to be shown on the webpage.
    '''

    @staticmethod
    def get_day_of_week(ai_model):
        if ai_model.problem_type == 1:  # classification
            data = DayofweekSentiment.objects.filter()
            terms = ['day_of_week', 'sentiment_actual_avg', 'sentiment_predicted_avg']
            terms_dict = {'day_of_week': ['sentiment_predicted_avg', 'sentiment_actual_avg']}
            title = 'Actual & estimated sentiment vs day of week'
            x_axis = "Hour"
            y_axis = "Sentiment"
        else:  # regression
            data = DayofweekRetweet.objects.filter()
            terms = ['day_of_week', 'retweet_actual_avg', 'retweet_predicted_avg']
            terms_dict = {'day_of_week': ['retweet_predicted_avg', 'retweet_actual_avg']}
            title = 'Actual & estimated retweet count vs day of week'
            x_axis = "Day of week"
            y_axis = "Retweet count"

        # Create data structure for charts
        data = DataPool(
            series=[{
                'options': {
                    'source': data.order_by('day_of_week')},
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
                    'labels': {
                        'step': '10',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    '''
    This function returns the hour versus predicted and actual sentiment (classification)
    or the hour versus actual retweet count and predicted retweet count (regression).
    Args:
      data: the data used to build the chart.
      ai_name (model): the problem type.
    Returns:
      cht (Chart): the resulting graph to be shown on the webpage.
    '''

    @staticmethod
    def get_hourly(ai_model):
        if ai_model.problem_type == 1:  # classification
            data = HourSentiment.objects.filter()
            terms = ['hour', 'sentiment_actual_avg', 'sentiment_predicted_avg']
            terms_dict = {'hour': ['sentiment_predicted_avg', 'sentiment_actual_avg']}
            title = 'Actual & estimated sentiment vs hour'
            x_axis = "Hour"
            y_axis = "Sentiment"
        else:  # regression
            data = HourRetweet.objects.filter()
            terms = ['hour', 'retweet_actual_avg', 'retweet_predicted_avg']
            terms_dict = {'hour': ['retweet_predicted_avg', 'retweet_actual_avg']}
            title = 'Actual & estimated retweet count vs hour'
            x_axis = "Hour"
            y_axis = "Retweet count"

        # Create data structure for charts
        data = DataPool(
            series=[{
                'options': {
                    'source': data.order_by('hour')},
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
                    'labels': {
                        'step': '10',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    '''
    This function returns the number of favorites versus predicted and actual sentiment (classification)
    or the number of favorites versus actual retweet count and predicted retweet count (regression).
    Args:
      data: the data used to build the chart.
      ai_name (model): the problem type.
    Returns:
      cht (Chart): the resulting graph to be shown on the webpage.
    '''

    @staticmethod
    def get_favorites_vs_x(data, ai_model):
        if ai_model.problem_type == 1:  # classification
            terms = ['favourites_count', 'sentiment_estimated', 'sentiment_actual']
            terms_dict = {'favourites_count': ['sentiment_estimated', 'sentiment_actual']}
            title = 'Actual & estimated sentiment vs favorites count'
            x_axis = "Favorites count"
            y_axis = "Sentiment"
        else:  # regression
            terms = ['favourites_count', 'retweet_count_estimated', 'retweet_count_actual']
            terms_dict = {'favourites_count': ['retweet_count_estimated', 'retweet_count_actual']}
            title = 'Actual & estimated retweet count vs favorites count'
            x_axis = "Favorites count"
            y_axis = "Retweet count"

        # Create data structure for charts
        data = DataPool(
            series=[{
                'options': {
                    'source': data.order_by('favourites_count')},
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
                    'labels': {
                        'step': '10',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    '''
    This function returns the follower count versus predicted and actual sentiment (classification)
    or the follower count versus actual retweet count and predicted retweet count (regression).
    Args:
      data: the data used to build the chart.
      ai_name (model): the problem type.
    Returns:
      cht (Chart): the resulting graph to be shown on the webpage.
    '''

    @staticmethod
    def get_followers_vs_x(data, ai_model):
        if ai_model.problem_type == 1:  # classification
            terms = ['followers_count', 'sentiment_estimated', 'sentiment_actual']
            terms_dict = {'followers_count': ['sentiment_estimated', 'sentiment_actual']}
            title = 'Actual & estimated sentiment vs followers count'
            x_axis = "Followers count"
            y_axis = "Sentiment"
        else:  # regression
            terms = ['followers_count', 'retweet_count_estimated', 'retweet_count_actual']
            terms_dict = {'followers_count': ['retweet_count_estimated', 'retweet_count_actual']}
            title = 'Actual & estimated retweet count vs followers count'
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
                    'labels': {
                        'step': '10',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht