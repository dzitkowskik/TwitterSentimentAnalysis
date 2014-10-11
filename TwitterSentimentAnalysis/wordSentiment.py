from config import Config
import re
import math
import sys
reload(sys)
sys.setdefaultencoding('utf-8')


class WordSentimentAnalyzer(object):
    def __init__(self, cfg_file_name):
        self.cfg = Config(file(cfg_file_name))
        self.__pattern_split = re.compile(r"\W+")
        self.__initialize()

    def __initialize(self):
        afinn_file = open(self.cfg.words_file)
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
