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
        words = self.__pattern_split.split(text.lower())
        sentiments = map(lambda word: self.words_dict.get(word, 0), words)
        if sentiments:
            sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
        else:
            sentiment = 0
        return sentiment
