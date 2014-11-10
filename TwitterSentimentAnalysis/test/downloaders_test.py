import unittest
from TwitterSentimentAnalysis import downloaders, core

class WordSentimentAnalyzerTestCase(unittest.TestCase):
    def setUp(self):
        core.initialize('TwitterSentimentAnalysis/test/test_configuration.cfg')
        # TODO

    def tearDown(self):
        core.terminate()
        # TODO

    def test_download_tweets_using_query(self):
        print "TODO"
        # TODO


if __name__ == '__main__':
    unittest.main()