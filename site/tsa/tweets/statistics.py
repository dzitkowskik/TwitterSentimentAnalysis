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
        """
        A method to return all possible choices for Statistic enum values

        :return: a list of tuples of all possible enum choices
        """
        members = inspect.getmembers(cls, lambda memb: not(inspect.isroutine(memb)))
        props = [m for m in members if not(m[0][:2] == '__')]
        return tuple([(p[0], p[1].value) for p in props])


class TweetStatistics(object):
    @staticmethod
    def get_chart(stat_type, data, ai_model):
        """
        A factory method for statistic which creates passed type of chart from passed data.
        :param stat_type: an object of StatisticEnum class as a type of statistic to create
        :param data: a data from which a chart should be created
        :param ai_model: An AI name used for creating passed data
        :return: an object of django_chartit chart
        """
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

    @staticmethod
    def save_data_for_statistics(data, ai_name):
        """
        This function saves the data from the database to a model used to make the graphs.
        Args:
          data: the dataset used to fill the model with data.
          ai_name (model): the saved artificial intelligence.
        """
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

            print hour_vs_sentiment
            print hour_vs_retweet_count
            print dayofweek_vs_sentiment
            print dayofweek_retweet_count
            print len(hour_vs_sentiment)
            print len(hour_vs_retweet_count)
            print len(dayofweek_vs_sentiment)
            print len(dayofweek_retweet_count)


        else:
            raise NameError('ai_name cannot be blank')

    @staticmethod
    def get_sample_chart(data, ai_model):
        """
        This function returns a sample chart.
        Args:
          data: the data used to build the chart.
          ai_name (model): the problem type.
        Returns:
          cht (Chart): the resulting graph to be shown on the webpage.
        """
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

        print data.values('hour').annotate(Avg('retweet_count_actual'), Avg('retweet_count_estimated'))

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

    @staticmethod
    def get_day_of_week(ai_model):
        """
        This function returns the day of the week versus predicted and actual sentiment (classification)
        or the day of the week versus actual retweet count and predicted retweet count (regression).
        Args:
          data: the data used to build the chart.
          ai_name (model): the problem type.
        Returns:
          cht (Chart): the resulting graph to be shown on the webpage.
        """
        if ai_model.problem_type == 1:  # classification
            data = DayofweekSentiment.objects.filter(ai=ai_model)
            terms = ['day_of_week', 'sentiment_actual_avg', 'sentiment_predicted_avg']
            terms_dict = {'day_of_week': ['sentiment_predicted_avg', 'sentiment_actual_avg']}
            title = 'Actual & estimated sentiment vs day of week'
            x_axis = "Day of the week"
            y_axis = "Sentiment"
        else:  # regression
            data = DayofweekRetweet.objects.filter(ai=ai_model)
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
                        'step': '5',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    @staticmethod
    def get_hourly(ai_model):
        """
        This function returns the hour versus predicted and actual sentiment (classification)
        or the hour versus actual retweet count and predicted retweet count (regression).
        Args:
          data: the data used to build the chart.
          ai_name (model): the problem type.
        Returns:
          cht (Chart): the resulting graph to be shown on the webpage.
        """
        print ai_model
        print HourSentiment.objects.filter(ai=ai_model).values()
        if ai_model.problem_type == 1:  # classification
            data = HourSentiment.objects.filter(ai=ai_model)
            terms = ['hour', 'sentiment_actual_avg', 'sentiment_predicted_avg']
            terms_dict = {'hour': ['sentiment_predicted_avg', 'sentiment_actual_avg']}
            title = 'Actual & estimated sentiment vs hour'
            x_axis = "Hour"
            y_axis = "Sentiment"
        else:  # regression
            data = HourRetweet.objects.filter(ai=ai_model)
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
                        'step': '5',
                        'maxStaggerLines': '1'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    @staticmethod
    def get_favorites_vs_x(data, ai_model):
        """
        This function returns the number of favorites versus predicted and actual sentiment (classification)
        or the number of favorites versus actual retweet count and predicted retweet count (regression).
        Args:
          data: the data used to build the chart.
          ai_name (model): the problem type.
        Returns:
          cht (Chart): the resulting graph to be shown on the webpage.
        """
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

        print data.values()
        print data.order_by('favourites_count').values()

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
                        'maxStaggerLines': '2'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht

    @staticmethod
    def get_followers_vs_x(data, ai_model):
        """
        This function returns the follower count versus predicted and actual sentiment (classification)
        or the follower count versus actual retweet count and predicted retweet count (regression).
        Args:
          data: the data used to build the chart.
          ai_name (model): the problem type.
        Returns:
          cht (Chart): the resulting graph to be shown on the webpage.
        """
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
                        'maxStaggerLines': '2'
                    },
                    'title': {
                        'text': x_axis}},
                'yAxis': {
                    'title': {
                        'text': y_axis}}})

        return cht