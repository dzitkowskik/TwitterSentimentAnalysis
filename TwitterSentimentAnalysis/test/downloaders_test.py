import uuid
import unittest
from TwitterSentimentAnalysis import downloaders, core


class DownloadTweetsTestCase(unittest.TestCase):
    def setUp(self):
        core.initialize('TwitterSentimentAnalysis/test/test_configuration.cfg')
        self.tweet_downloader = downloaders.TweetDownloader()
        self.test_db = self.tweet_downloader.db
        self.test_table_name = "tweet_download_" + uuid.uuid4().hex + "_test"

    def tearDown(self):
        core.terminate()
        self.test_db.drop_collection(self.test_table_name)

    def test_download_tweets_using_query(self):
        expected = 50
        self.tweet_downloader.download_tweets_using_query("erasmus", expected, self.test_table_name, tag="test")
        number_of_tweets_in_db = self.test_db[self.test_table_name].count()
        self.assertEqual(number_of_tweets_in_db, expected)
        tweet = self.test_db[self.test_table_name].find_one()
        self.assertRegexpMatches(tweet["text"].lower(), "erasmus")


if __name__ == '__main__':
    unittest.main()