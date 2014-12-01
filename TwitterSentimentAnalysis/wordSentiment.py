# Karol Dzitkowski
# k.dzitkowski@gmail.com
# 10-10-2014
__author__ = 'Karol Dzitkowski'

import inject
from config import Config
import re
import math
import core


class WordSentimentAnalyzer(object):
    """Provides a functionality of computing a sentiment from a text based on words

    This class is based on AFINN - an affective lexicon by Finn Arup Nielsen
    (originally available at https://gist.github.com/1035399) License: GPLv3
    """

    @inject.params(config=Config)
    def __init__(self, config):
        self.__pattern_split = re.compile(r"\W+")
        self.cfg = config
        self.__initialize()

    def __initialize(self):
        afinn_file = open(core.convert_rel_to_absolute(self.cfg.words_file))
        words = [ws.strip().split('\t') for ws in afinn_file]
        pairs = lambda (word, sent): (word, int(sent))
        self.words_dict = dict(map(pairs, words))

    def get_word_sentiment(self, text):
        """Calculates sentiment as a function of word weights contained in a file

        Args:
          text (string): A text to analize in terms of a sentiment

        Returns: A word sentiment grade
        """
        words = self.__pattern_split.split(text.lower())
        sentiments = map(lambda word: self.words_dict.get(word, 0), words)
        if sentiments:
            sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
        else:
            sentiment = 0
        return sentiment
