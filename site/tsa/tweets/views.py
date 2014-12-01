# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

from django.shortcuts import render
from django.views.generic import View
import inject
from pymongo import MongoClient
from config import Config
from tweepy import Cursor
from TwitterSentimentAnalysis import core
from TwitterSentimentAnalysis.datasets import DatasetFactory
from TwitterSentimentAnalysis.ai import AIEnum, AI
from forms import QueryForm, AnalysisForm, ActionEnum, StatisticsForm
from TwitterSentimentAnalysis.downloaders import TweetDownloader
from statistics import TweetStatistics
from statistics import StatisticEnum
from models import ArtificialIntelligence, Tweet


class TweetSearchView(View):
    """This view is used for querying the tweets from twitter

    It provides an interface for searching and downloading sets of tweets, saving it in db with some tag
    in order to perform some classification and regression analysis later
    """
    template_name = "search.html"
    language = 'en'
    tweets_per_page = 10
    pages_shown_count = 5
    max_pages = 1000

    @inject.params(config=Config)
    def __init__(self, config, **kwargs):
        super(TweetSearchView, self).__init__(**kwargs)
        self.cfg = config
        self.api = core.get_tweepy_api(self.cfg)
        self.td = TweetDownloader()

    def get(self, request):
        """A handler for a GET web request from search view

        Provides a web page with unfilled query form and listed tweets from user timeline
        :param request: HTTP web GET request
        :return: a web page with a search interface
        """
        records = self.api.home_timeline(count=self.tweets_per_page)
        header = 'Home timeline'
        form = QueryForm()
        pages, page = self.__get_pages_range(form)
        context = {'tweets': records, 'form': form, 'header': header, 'pages': pages}
        return render(request, self.template_name, context)

    def post(self, request):
        """A handler for a POST web request from search view

        Provides a web page with filled query form and listed queried tweets from twitter and/or
        saved queried data in mongoDB database with a specified tag
        :param request: HTTP web POST request
        :return: a web page with a search interface
        """
        form = QueryForm(request.POST)
        if form.is_valid():
            if 'form_save' in request.POST:
                # TODO: Implement async waiting for saving with progress bar
                self.save(form)
            return self.search(request, form)

        header = 'Error occurred'
        pages = []
        return render(
            request,
            self.template_name,
            {'form': form, 'header': header, 'pages': pages})

    def search(self, request, form):
        """Provide search functionality for post handler"""
        query = form.cleaned_data['query']
        pages, page = self.__get_pages_range(form)

        if not query:
            header = 'Home timeline'
            records = self.api.home_timeline(count=self.tweets_per_page, page=page)
        else:
            header = "Search: " + query
            p = Cursor(self.api.search, q=query, lang=self.language, rpp=self.tweets_per_page).pages(page)
            records = None
            for records in p:
                pass
        if records:
            context = {'tweets': records, 'form': form, 'header': header, 'pages': pages}
            return render(request, self.template_name, context)
        else:
            header = 'Nothing found!'
            pages = []
            return render(request, self.template_name, {'form': form, 'header': header, 'pages': pages})

    def save(self, form):
        """Provide save functionality for post handler"""
        query = form.cleaned_data['query']
        name = form.cleaned_data['name']
        limit = form.cleaned_data['limit']
        undersample = form.cleaned_data['undersample']
        self.td.download_tweets_using_query(
            query,
            limit,
            self.cfg.test_tweets_table,
            tag=name,
            undersample=undersample)

    def __get_pages_range(self, form):
        page = None
        if 'cleaned_data' in form and 'page' in form.cleaned_data:
            page = form.cleaned_data['page']
        if page is None:
            page = 1
        start = max(1, page - (self.pages_shown_count / 2))
        end = start + self.pages_shown_count
        return range(start, end + 1), page


class AnalysisView(View):
    """This view is used for analysis of the tweets downloaded twitter or predefined stored in db

    It provides an interface for running an AI algorithms on tweet sets (regression and classification) and
    displays results as well as calculated error rate. It can also be used to save the results of analysis in db.
    """
    template_name = "analysis.html"
    default_header = "Data analysis"
    tweets_per_page = 10
    pages_shown_count = 5
    max_pages = 1000

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client, **kwargs):
        super(AnalysisView, self).__init__(**kwargs)
        self.cfg = config
        self.db = db_client[config.db_database_name]

    def get(self, request):
        """A handler for a GET web request from analysis view

        Provides a web page with unfilled analysis form with a blank list of analyzed tweets
        :param request: HTTP web GET request
        :return: a web page with a analyze interface
        """
        form = self.get_form()
        context = {'header': self.default_header, 'form': form}
        return render(request, self.template_name, context)

    def post(self, request):
        """A handler for a POST web request from analysis view

        Provides a web page with filled analysis form and listed tweets with actual and predicted values
        of attributes using some AI, and shows an error rate of classification or regression. It provides
        also a functionality to save these results with some custom name. Analysis can be performed or on
        custom data or predefined stored in table train_data in mongoDB
        :param request: HTTP web POST request
        :return: a web page with a search interface
        """
        form = self.get_form(request.POST)
        if form.is_valid():
            ai, create = self.get_ai(form)
            ds, data, tag = self.get_data(form, ai, create)
            ds_train, ds_test = ds.splitWithProportion(0.5)

            if create:
                error = ai.run(ds_train, ds_test)
            else:
                error = ai.test(ds_test)

            ai.fill_with_predicted_data(ds, data)

            if form.cleaned_data['save_results']:
                name = form.cleaned_data['name']
                self.save_trained_ai(ai, name, tag)
                TweetStatistics.save_data_for_statistics(data, name)

            context = {
                'header': self.default_header,
                'form': form,
                'error': error,
                'data': data
            }
            return render(request, self.template_name, context)

        header = 'Error occurred'
        context = {'header': header, 'form': form}
        return render(request, self.template_name, context)

    def get_form(self, post=None):
        """Returns a analysis form with filled dropdown choices"""
        sets = self.get_tweet_sets()
        ais = map(lambda ai: (ai.name, ai.name), ArtificialIntelligence.objects.all())
        if post is None:
            return AnalysisForm(sets, ais)
        else:
            return AnalysisForm(sets, ais, post)

    # noinspection PyMethodMayBeStatic
    def get_ai(self, form):
        """Returns a AI object which was choosed using web interface

        :param form: a valid analysis form
        :return: tuple of AI object and boolean (true if create action, false if load)
        """
        action = int(form.cleaned_data['action'])
        if action == ActionEnum.Create.value:
            result = True
            ai_type = AIEnum[form.cleaned_data['ai_types']]
            ai = AI.factory(ai_type)
        else:
            result = False
            saved_ai_name = form.cleaned_data['saved_ais']
            ai_model = ArtificialIntelligence.objects.get(name=saved_ai_name)
            ai = AI.factory(AIEnum(ai_model.ai_type))
            ai.load(ai_model.path)
            form.cleaned_data['tweet_sets'] = ai_model.tag
        return ai, result

    def get_data(self, form, ai, create=True):
        """Provides a data stored in db choosed using web interface

        :param form: a valid analysis form
        :param ai: choosed AI object
        :param create: to create or to load AI model
        :return: tuple of dataset, data and a tag
        """
        problem_type, _ = ai.get_type()
        factory = DatasetFactory.factory(problem_type)
        custom = form.cleaned_data['custom_tweet_set']
        if custom or not create:
            tag = form.cleaned_data['tweet_sets']
            ds = factory.get_dataset(
                table_name=self.cfg.test_tweets_table,
                search_params={"isActive": True, "tag": tag})
            data = list(factory.get_data(
                table_name=self.cfg.test_tweets_table,
                search_params={"isActive": True, "tag": tag}))
        else:
            tag = ""
            ds = factory.get_dataset()
            data = list(factory.get_data())
        return ds, data, tag

    def get_tweet_sets(self):
        """Gets a list of distinct tweet tags that are available to analyze

        :return: return distinct tags in tweets table in mongoDB
        """
        table = self.db[self.cfg.test_tweets_table]
        tags = table.distinct('tag')
        result = []
        for tag in tags:
            result.append((tag, tag))
        return result

    def save_trained_ai(self, ai, name, tag):
        """Saves trained AI to a file and all necessary information to db used by django

        :param ai: AI object to store
        :param name: a name of AI we want to save
        :param tag: a tag of data that AI was trained for
        :raises: NameError
        """
        if name != "" and name is not None:
            save_path = core.convert_rel_to_absolute(
                self.cfg.ai_save_dir + name + ".ai")
            ai.save(save_path)
            problem_type, ai_type = ai.get_type()
            ai_model = ArtificialIntelligence(
                tag=tag,
                name=name,
                path=save_path,
                ai_type=ai_type.value,
                problem_type=problem_type.value)
            ai_model.save()
        else:
            raise NameError('Name cannot be blank')


class StatisticsView(View):
    """This view is used for showing charts of the analyzed tweet sets

    It provides an interface for browsing charts generated from actual and predicted data of tweets
    """
    template_name = "statistics.html"
    default_header = "Twitter sentiment statistics"

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client, **kwargs):
        super(StatisticsView, self).__init__(**kwargs)
        self.cfg = config
        self.db = db_client[config.db_database_name]

    def get(self, request):
        """A handler for a GET web request from statistic view

        Provides a web page with unfilled statistic form with a sample chart from stored data
        :param request: HTTP web GET request
        :return: a web page with a statistic interface
        """
        cht = TweetStatistics.get_chart(
            StatisticEnum.sample,
            Tweet.objects.all(),
            ArtificialIntelligence.objects.all()[:1].get())

        form = StatisticsForm()

        context = {
            'header': self.default_header,
            'chart_list': cht,
            'form': form}

        return render(request, self.template_name, context)

    def post(self, request):
        """A handler for a POST web request from statistic view

        Provides a web page with filled statistic form and shows generated chart
        :param request: HTTP web POST request
        :return: a web page with a statistic interface
        """
        form = StatisticsForm(request.POST)
        if form.is_valid():
            ai = form.cleaned_data['tweet_sets']
            data = Tweet.objects.filter(ai=ai)
            stat = StatisticEnum[form.cleaned_data['statistic_types']]
            cht = TweetStatistics.get_chart(stat, data, ai)
            context = {
                'header': self.default_header,
                'chart_list': cht,
                'form': form}
        else:
            cht = None
            form = StatisticsForm()
            header = 'Error occured'
            context = {'header': header, 'chart_list': cht, 'form': form}
        return render(request, self.template_name, context)


def contact(request):
    """Simple contact view page"""
    return render(request, 'contact.html', {})