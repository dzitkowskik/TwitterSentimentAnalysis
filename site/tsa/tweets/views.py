from django.shortcuts import render
from django.views.generic import View
import inject
from TwitterSentimentAnalysis import core
from config import Config
from forms import QueryForm
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

    def get(self, request):
        records = self.api.home_timeline(count=self.tweets_per_page)
        header = 'Home timeline'
        pages = self.__get_pages_range()
        context = {'tweets': records, 'form': QueryForm(), 'header': header, 'pages': pages}

        return render(request, self.template_name, context)

    def post(self, request):
        form = QueryForm(request.POST)
        if form.is_valid():
            if 'form_search' in request.POST:
                return self.search(request, form)
            elif 'form_save' in request.POST:
                return self.save(request, form)

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

    def save(self, request, form):
        query = form.cleaned_data['query']
        name = form.cleaned_data['name']
        limit = form.cleaned_data['limit']
        td = TweetDownloader()
        td.download_tweets_using_query(query, limit, 'test_data', tag=name)

        # TODO: Implement async waiting for saving with progress bar

        return self.search(request, form)

    def __get_pages_range(self, actual=1):
        if actual is None:
            actual = 1
        start = max(1, actual - (self.pages_shown_count / 2))
        end = start + self.pages_shown_count
        return range(start, end + 1)


def contact(request):
    return render(request, 'contact.html', {})