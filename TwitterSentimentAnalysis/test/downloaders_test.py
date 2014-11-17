import uuid
import unittest
from TwitterSentimentAnalysis import downloaders, core
from config import Config
import os


class DownloadTweetsTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        base_dir = os.path.dirname(__file__)
        configuration_file_path = os.path.join(base_dir, 'test_configuration.cfg')
        core.initialize(configuration_file_path)

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
        expected = 10
        self.tweet_downloader.download_tweets_using_query("erasmus", expected, self.test_table_name, tag="test")
        number_of_tweets_in_db = self.test_db[self.test_table_name].count()
        self.assertEqual(number_of_tweets_in_db, expected)

    def test_download_tweets_using_query_check_text_not_empty(self):
        self.tweet_downloader.download_tweets_using_query("erasmus", 1, self.test_table_name, tag="test")
        tweet = self.test_db[self.test_table_name].find_one()
        self.assertIsNotNone(tweet['text'])
        self.assertNotEqual(tweet['text'], "")

    def test_download_tweets_using_query_empty_query_check_equal_timeline(self):
        limit = 10
        query = ""
        self.tweet_downloader.download_tweets_using_query(query, limit, self.test_table_name, tag="test")
        cfg = Config(core.configuration_file_path)
        api = core.get_tweepy_api(cfg)
        expected = api.user_timeline(count=limit)[0].text
        actual = self.test_db[self.test_table_name].find_one({'text': expected})
        self.assertIsNotNone(actual)

    def test_download_tweets_using_query_empty_query_check_tweet_number(self):
        expected = 5
        query = ""
        self.tweet_downloader.download_tweets_using_query(query, expected, self.test_table_name, tag="test")
        actual = self.test_db[self.test_table_name].count()
        cfg = Config(core.configuration_file_path)
        api = core.get_tweepy_api(cfg)
        expected = min(len(api.user_timeline(count=expected)), expected)
        self.assertEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()