from django.apps import AppConfig
from TwitterSentimentAnalysis import core


class TweetsAppConfig(AppConfig):
    """
    This class is used to perform an TweetSentimentAnalysis
    initialization before using it in tweets app in web site
    """
    name = 'tweets'
    verbose_name = "TweetsAppConfig"

    def ready(self):
        """Initializes TweetSentimentAnalysis package"""
        core.initialize()
