from django.shortcuts import render
from django.views.generic import View
import inject
from TwitterSentimentAnalysis import core
from config import Config


# noinspection PyMethodMayBeStatic
class TweetSearchView(View):
    @inject.params(config=Config)
    def __init__(self, config):
        self.cfg = config
        self.api = core.get_tweepy_api(self.cfg, False)

    def get(self, request):
        records = self.api.home_timeline()[:10]
        context = {'tweets': records}
        template = "home.html"
        return render(request, template, context)
