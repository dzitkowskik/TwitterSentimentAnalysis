import unittest
from TwitterSentimentAnalysis import downloaders, core

class DownloadTweetsTestCase(unittest.TestCase):
    def setUp(self):
        core.initialize('TwitterSentimentAnalysis/test/test_configuration.cfg')
        self.tweet_downloader = downloaders.TweetDownloader()
        self.test_db = self.tweet_downloader.db
        # TODO

    def tearDown(self):
        core.terminate()
        self.test_db.drop_collection("tweet_download_test_db")

    def test_download_tweets_using_query(self):
        expected = 50
        self.tweet_downloader.download_tweets_using_query("erasmus", expected, "tweet_download_test_db", tag="test")
        number_of_tweets_in_db = self.test_db.tweet_download_test_db.count()
        self.assertEqual(number_of_tweets_in_db, expected)
        tweet = self.test_db.tweet_download_test_db.find_one()
        self.assertRegexpMatches(tweet["text"].lower(), "erasmus")


if __name__ == '__main__':
    unittest.main()