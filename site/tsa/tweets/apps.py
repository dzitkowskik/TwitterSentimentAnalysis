import os
from django.apps import AppConfig
from TwitterSentimentAnalysis import core


class TweetsAppConfig(AppConfig):
    name = 'tweets'
    verbose_name = "TweetsAppConfig"

    def ready(self):
        base_dir = os.path.dirname(os.path.dirname(__file__))
        core.configuration_file_path = os.path.join(base_dir, '../../TwitterSentimentAnalysis/configuration.cfg')
        print core.configuration_file_path
        core.initialize()
