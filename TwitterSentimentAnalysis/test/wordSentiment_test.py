import unittest
import math
from TwitterSentimentAnalysis import wordSentiment, core


class WordSentimentAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        core.initialize('TwitterSentimentAnalysis/test/test_configuration.cfg')
        words_dict = {'great': 3, 'noob': -2, 'super': 1, 'bad': -1}
        self.sentiment_analyzer = wordSentiment.WordSentimentAnalyzer()
        self.sentiment_analyzer.words_dict = words_dict

    def tearDown(self):
        core.terminate()

    def test_get_word_sentiment(self):
        result = self.sentiment_analyzer.get_word_sentiment('great job')
        expected = 3 / math.sqrt(2)
        self.assertEqual(result, expected)


if __name__ == '__main__':
    unittest.main()
