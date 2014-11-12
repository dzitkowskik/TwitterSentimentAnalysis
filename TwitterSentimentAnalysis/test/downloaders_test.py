import uuid
import unittest
from TwitterSentimentAnalysis import downloaders, core


class DownloadTweetsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        core.initialize('TwitterSentimentAnalysis/test/test_configuration.cfg')

    @classmethod
    def tearDownClass(cls):
        core.terminate()

    def setUp(self):
        self.tweet_downloader = downloaders.TweetDownloader()
        self.test_db = self.tweet_downloader.db
        self.test_table_name = "tweet_download_" + uuid.uuid4().hex + "_test"

    def tearDown(self):
        self.test_db.drop_collection(self.test_table_name)

    def test_download_tweets_using_query_check_tweet_number(self):
        expected = 50
        self.tweet_downloader.download_tweets_using_query("erasmus", expected, self.test_table_name, tag="test")
        number_of_tweets_in_db = self.test_db[self.test_table_name].count()
        self.assertEqual(number_of_tweets_in_db, expected)

    def test_download_tweets_using_query_check_text_not_empty(self):
        self.tweet_downloader.download_tweets_using_query("erasmus", 1, self.test_table_name, tag="test")
        tweet = self.test_db[self.test_table_name].find_one()
        self.assertIsNotNone(tweet['text'])
        self.assertNotEqual(tweet['text'], "")

if __name__ == '__main__':
    unittest.main()