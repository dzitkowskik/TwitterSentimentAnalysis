from django.shortcuts import render
from django.views.generic import View
import inject
from TwitterSentimentAnalysis import core
from config import Config
from forms import QueryForm


class TweetSearchView(View):
    template_name = "search.html"
    language = 'en'
    tweets_per_page = 10
    pages_shown_count = 5

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
            query = form.cleaned_data['query']
            page = form.cleaned_data['page']
            pages = self.__get_pages_range(page)
            records = self.api.search(query, self.language, rpp=self.tweets_per_page, page=page)
            header = "Search: " + query
            context = {'tweets': records, 'form': form, 'header': header, 'pages': pages}
            return render(request, self.template_name, context)

        header = 'Nothing found'
        pages = []
        return render(request, self.template_name, {'form': form, 'header': header, 'pages': pages})

    def __get_pages_range(self, actual=1):
        start = max(1, actual - (self.pages_shown_count / 2))
        end = start + self.pages_shown_count
        return range(start, end + 1)

def contact(request):
    return render(request, 'contact.html', {})