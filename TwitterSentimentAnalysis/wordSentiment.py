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
    """The summary line for a class docstring should fit on one line.

    If the class has public attributes, they should be documented here
    in an ``Attributes`` section and follow the same formatting as a
    function's ``Args`` section.

    Attributes:
      attr1 (str): Description of `attr1`.
      attr2 (list of str): Description of `attr2`.
      attr3 (int): Description of `attr3`.

    """
    @inject.params(config=Config)
    def __init__(self, config):
        """Example of docstring on the __init__ method.

        The __init__ method may be documented in either the class level
        docstring, or as a docstring on the __init__ method itself.

        Either form is acceptable, but the two should not be mixed. Choose one
        convention to document the __init__ method and be consistent with it.

        Note:
          Do not include the `self` parameter in the ``Args`` section.

        Args:
          param1 (str): Description of `param1`.
          param2 (list of str): Description of `param2`. Multiple
            lines are supported.
          param3 (int, optional): Description of `param3`, defaults to 0.

        """
        self.__pattern_split = re.compile(r"\W+")
        self.cfg = config
        self.__initialize()

    def __initialize(self):
        afinn_file = open(core.convert_rel_to_absolute(self.cfg.words_file))
        words = [ws.strip().split('\t') for ws in afinn_file]
        pairs = lambda (word, sent): (word, int(sent))
        self.words_dict = dict(map(pairs, words))

    def get_word_sentiment(self, text):
        """Class methods are similar to regular functions.

        Note:
          Do not include the `self` parameter in the ``Args`` section.

        Args:
          param1: The first parameter.
          param2: The second parameter.

        Returns:
          True if successful, False otherwise.

        """
        words = self.__pattern_split.split(text.lower())
        sentiments = map(lambda word: self.words_dict.get(word, 0), words)
        if sentiments:
            sentiment = float(sum(sentiments))/math.sqrt(len(sentiments))
        else:
            sentiment = 0
        return sentiment
