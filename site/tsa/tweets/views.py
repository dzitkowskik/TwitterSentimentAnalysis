from django.shortcuts import render
from django.views.generic import View
import inject
from TwitterSentimentAnalysis import core
from config import Config
from forms import QueryForm


class TweetSearchView(View):
    template_name = "search.html"
    language = 'en'

    @inject.params(config=Config)
    def __init__(self, config):
        self.cfg = config
        self.api = core.get_tweepy_api(self.cfg)

    def get(self, request):
        records = self.api.home_timeline()
        header = 'Home timeline'
        context = {'tweets': records, 'form': QueryForm(), 'header': header}

        return render(request, self.template_name, context)

    def post(self, request):
        form = QueryForm(request.POST)
        if form.is_valid():
            query = form.cleaned_data['query']
            records = self.api.search(query, self.language)
            header = "Search: " + query
            context = {'tweets': records, 'form': form, 'header': header}
            return render(request, self.template_name, context)

        header = 'Nothing found'
        return render(request, self.template_name, {'form': form, 'header': header})


def contact(request):
    return render(request, 'contact.html', {})