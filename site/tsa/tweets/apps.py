import os
from django.apps import AppConfig
from TwitterSentimentAnalysis import core


class TweetsAppConfig(AppConfig):
    name = 'tweets'
    verbose_name = "TweetsAppConfig"

    def ready(self):
        core.initialize()
