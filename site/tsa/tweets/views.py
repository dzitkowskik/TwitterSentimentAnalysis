import os
from django.shortcuts import render
from django.views.generic import View
import inject
from pymongo import MongoClient
from TwitterSentimentAnalysis import core
from config import Config
from TwitterSentimentAnalysis.datasets import DatasetFactory
from TwitterSentimentAnalysis.neuralNetworks import AIEnum, NeuralNetwork
from forms import QueryForm, AnalysisForm, ActionEnum
from tweepy import Cursor
from TwitterSentimentAnalysis.downloaders import TweetDownloader


class TweetSearchView(View):
    template_name = "search.html"
    language = 'en'
    tweets_per_page = 10
    pages_shown_count = 5
    max_pages = 1000

    @inject.params(config=Config)
    def __init__(self, config):
        self.cfg = config
        self.api = core.get_tweepy_api(self.cfg)
        self.td = TweetDownloader()

    def get(self, request):
        records = self.api.home_timeline(count=self.tweets_per_page)
        header = 'Home timeline'
        pages = self.__get_pages_range()
        context = {'tweets': records, 'form': QueryForm(), 'header': header, 'pages': pages}

        return render(request, self.template_name, context)

    def post(self, request):
        form = QueryForm(request.POST)
        if form.is_valid():
            if 'form_save' in request.POST:
                # TODO: Implement async waiting for saving with progress bar
                self.save(form)
            return self.search(request, form)

        header = 'Error occurred'
        pages = []
        return render(request, self.template_name, {'form': form, 'header': header, 'pages': pages})

    def search(self, request, form):
        query = form.cleaned_data['query']
        page = form.cleaned_data['page']
        pages = self.__get_pages_range(page)

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
        query = form.cleaned_data['query']
        name = form.cleaned_data['name']
        limit = form.cleaned_data['limit']
        self.td.download_tweets_using_query(query, limit, self.cfg.test_tweets_table, tag=name)

    def __get_pages_range(self, actual=1):
        if actual is None:
            actual = 1
        start = max(1, actual - (self.pages_shown_count / 2))
        end = start + self.pages_shown_count
        return range(start, end + 1)


class AnalysisView(View):
    template_name = "analysis.html"
    tweets_per_page = 10
    pages_shown_count = 5
    max_pages = 1000

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    def get_form(self, post=None):
        sets = self.get_tweet_sets()
        ais = self.get_saved_ais()
        if post is None:
            return AnalysisForm(sets, ais)
        else:
            return AnalysisForm(sets, ais, post)

    def get(self, request):
        header = "Twitter sentiment analysis"
        form = self.get_form()
        context = {'header': header, 'form': form}
        return render(request, self.template_name, context)

    def post(self, request):
        form = self.get_form(request.POST)
        if form.is_valid():
            action = form.cleaned_data['action']
            save = form.cleaned_data['save_results']
            custom = form.cleaned_data['custom_tweet_set']

            tag = form.cleaned_data['tweet_sets']
            ai_type = AIEnum[form.cleaned_data['ai_types']]

            network = NeuralNetwork.factory(ai_type)
            problem_type = network.get_type()
            factory = DatasetFactory.factory(problem_type)
            ds = factory.get_dataset(
                table_name='test_data',
                search_params={"isActive": True, "tag": tag})
            ds_train, ds_test = ds.splitWithProportion(0.5)
            error = network.run(ds_train, ds_test)

            header = "Twitter sentiment analysis"
            context = {'header': header, 'form': form, 'error': error}
            return render(request, self.template_name, context)

        header = 'Error occurred'
        context = {'header': header, 'form': form}
        return render(request, self.template_name, context)

    def get_tweet_sets(self):
        table = self.db[self.cfg.test_tweets_table]
        tags = table.distinct('tag')
        result = []
        for tag in tags:
            result.append((tag, tag))
        return result

    def get_saved_ais(self):
        results = []
        save_dir = core.convert_rel_to_absolute(self.cfg.ai_save_dir)
        for file_name in os.listdir(save_dir):
            if file_name.endswith(".net"):
                results.append((file_name, file_name))
        return results

    def get_predicted_data(self):
        pass

    def save_trained_network(self):
        pass


class StatisticsView(View):
    template_name = "statistics.html"

    @inject.params(config=Config, db_client=MongoClient)
    def __init__(self, config, db_client):
        self.cfg = config
        self.db = db_client[config.db_database_name]

    def get(self, request):
        header = "Twitter sentiment statistics"
        context = {'header': header}
        return render(request, self.template_name, context)


def contact(request):
    return render(request, 'contact.html', {})